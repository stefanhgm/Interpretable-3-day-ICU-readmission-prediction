import argparse

import re
import time
from datetime import datetime

import json

import numpy as np
import os
import pandas as pd
import tensorflow as tf
import random as python_random
from joblib import Parallel, delayed
from pandas.core.dtypes.common import is_string_dtype
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from xgboost import XGBClassifier
from tensorflow import keras
from tensorflow.keras import layers

from experiments.custom_ebm.ebm_enable_unk_min_samples_bin import ExplainableBoostingClassifier
from experiments.exp1_LR_EBM_parameter_tuning_all_features import evaluate_model, \
    add_categorical_dummies_and_nan_indicators, plot_performance, best_1d_ebm_risk_functions, best_ebm_params_2d_model, \
    best_2d_ebm_risk_functions, best_2d_bin_size, removed_risk_functions, best_lr_features, z_normalize_based_on_train, \
    best_lr_params_130_features
from experiments.variables import seed, saps_scoring_functions
from helper.io import read_item_processing_descriptions_from_excel
from helper.util import read_features_years_labels, create_splits, get_split, output_wrapper, get_pid_case_ids_map
from preprocessing.step5_generate_features_for_stays import decode_feature_time_span, timeseries_imputation
from preprocessing.step5b_generate_additional_features_for_stays import static_is_3d_icu_readmission, \
    static_icu_station, static_icu_length_of_stay_days, static_length_of_stay_before_icu_days
from research_database.research_database_communication import ResearchDBConnection

# Resource used for PT: https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html
# Added two features for model complexity (max_depth, min_child_weight) and one for robustness (learning_rate).
gbm_params = {
    'learning_rate': [i * 0.02 for i in range(1, 11)],  # Default 0.3, step size shrinkage against over-fitting.
    'min_child_weight': [1, 3, 5, 7, 9],  # Default 1, min sum of instance weight (hessian) needed in a child.
    'max_depth': [1, 2, 3, 4, 5],  # Default 6, maximum depth of a tree. For 4-5 already over-fitting.
    'objective': ['binary:logistic'],  # Output probability.
    'n_estimators': [100, 500],  # Default 100.
    'eval_metric': ['aucpr'],
    'use_label_encoder': [False],  # Hide deprecation warning.
    'seed': [seed],
}

best_gbm_params_full_features = {
    'learning_rate': 0.02,
    'min_child_weight': 7,
    'max_depth': 3,
    'objective': 'binary:logistic',
    'n_estimators': 500,
    'eval_metric': 'aucpr',
    'use_label_encoder': False,
    'seed': seed
}

rnn_params = {
    'lstm_neurons': [32, 64],
    'dense_neurons': [64, 128],
    'dropout': [0., 0.1, 0.2, 0.3, 0.4, 0.5],  # default 0.0 (tune 0, 0.05, 0.1, 0.15, 0.2, 0.25)
    'recurrent_dropout': [0.],  # default 0.0
    'learning_rate': [0.1, 0.01, 0.001, 0.0001, 0.00001],  # default 0.001 (tune 0.1, 0.01, 0.001, 0.0001, 0.00001)
    'batch_size': [32],  # default None, i.e. 32
    'rnn_layers': [1],
    'epochs': [10],  # default 1
    'seed': [seed]
}

best_rnn_params_full_features = {
    'lstm_neurons': 64,
    'dense_neurons': 64,
    'dropout': 0.4,
    'recurrent_dropout': 0.,
    'learning_rate': 0.001,
    'batch_size': 32,
    'rnn_layers': 1,
    'epochs': 3,
    'seed': seed
}


def main():
    parser = argparse.ArgumentParser(description='4. experiment: performance comparison with SAPS II/GBM/LSTM-RNN.')
    parser.add_argument('password', type=str, help='Database password.')
    parser.add_argument('item_overview', type=str, help='Description of all PDMS items and generated variables.')
    parser.add_argument('--data', type=str, help='CSV file containing features and targets.')
    parser.add_argument('--data_rnn', type=str, help='CSV file containing features and targets for RNN.')
    parser.add_argument('--figures_output_path', type=str, help='Directory where generated figures will be stored.')
    parser.add_argument('--model', type=str, help='Model used for parameter tuning (saps/gbm/rnn/rnn-features).')
    parser.add_argument('--parameter_tuning', action='store_true', help='Use reduced set of features.')
    parser.add_argument('--variable_names', nargs='+', help='Only convert specified variables.', default='')
    args, _ = parser.parse_known_args()

    assert args.model in ['saps', 'gbm', 'rnn', 'rnn-features', 'all']

    if args.model == 'all':
        performance_evaluations = []
        # EBM
        print(f"Evaluate EBM model.")
        feature_columns, year_column, label_column, data = read_features_years_labels(args.data)
        removed_feature_columns = [f for f in feature_columns if f not in best_1d_ebm_risk_functions]
        data.drop(removed_feature_columns, axis=1, inplace=True)
        data = data[best_1d_ebm_risk_functions + [year_column] + [label_column]]
        feature_columns = best_1d_ebm_risk_functions
        assert len(feature_columns) == len(data.columns.tolist()) - 2
        assert len(feature_columns) == 80

        splits = create_splits(data, feature_columns, year_column, label_column)
        train_x, train_y, valid_x, valid_y, test_x, test_y = \
            get_split(splits.iloc[-1], data, feature_columns, year_column, label_column)
        clf = ExplainableBoostingClassifier()
        clf.set_params(**{**best_ebm_params_2d_model, 'interactions': best_2d_ebm_risk_functions,
                          'max_interaction_bins': best_2d_bin_size})
        clf.fit(train_x, train_y)
        print(f"\tFull split scores: {evaluate_model(clf, train_x, train_y, valid_x, valid_y, test_x, test_y)}")
        print(f"\tSet {len(removed_risk_functions)} removed risk functions to zero.")
        # Set excluded RF to zero.
        for rf in removed_risk_functions:
            clf.additive_terms_[clf.feature_names.index(rf)].fill(0)
        print(f"\tFull split scores: {evaluate_model(clf, train_x, train_y, valid_x, valid_y, test_x, test_y)}")

        # Store model.
        # store_ebm_model(clf, get_timestamp(), args.password)

        held_out = (test_y, clf.predict_proba(test_x)[:, 1])
        temporal_splits = []
        for i in range(0, 5):
            train_x, train_y, valid_x, valid_y, test_x, test_y = \
                get_split(splits.iloc[i], data, feature_columns, year_column, label_column)
            clf = ExplainableBoostingClassifier()
            clf.set_params(**{**best_ebm_params_2d_model, 'interactions': best_2d_ebm_risk_functions,
                              'max_interaction_bins': best_2d_bin_size})
            clf.fit(train_x, train_y)
            # Set excluded RF to zero.
            for rf in removed_risk_functions:
                clf.additive_terms_[clf.feature_names.index(rf)].fill(0)
            print(f"\t\tTemp split scores: {evaluate_model(clf, train_x, train_y, valid_x, valid_y, test_x, test_y)}")
            temporal_splits.append((test_y, clf.predict_proba(test_x)[:, 1]))
        performance_evaluations.append((held_out, temporal_splits, 'EBM', 'blue'))

        # SAPS
        print(f"\nEvaluate SAPS II score.")
        feature_columns, year_column, label_column, data = read_features_years_labels(args.data)
        saps_features = list(saps_scoring_functions.keys())
        for variable in saps_features:
            data[variable] = data.apply(saps_scoring_functions[variable], axis=1)
            assert not any(data[variable].isna())
        data = data.drop(feature_columns, axis=1)
        feature_columns = saps_features
        assert data.shape[1] == len(saps_features) + 2
        # Create single feature for SAPS II score.
        data['SAPS II score'] = data[feature_columns].sum(axis=1)
        data = data.drop(saps_features, axis=1)
        feature_columns = ['SAPS II score']
        print(f"Mean SAPS II score: {data['SAPS II score'].sum() / data.shape[0]}")
        assert data.shape[1] == 3

        splits = create_splits(data, feature_columns, year_column, label_column)
        train_x, train_y, valid_x, valid_y, test_x, test_y = \
            get_split(splits.iloc[-1], data, feature_columns, year_column, label_column)
        # Simply use SAPS II score s predictions.
        print(f"\tFull split scores: {evaluate_model(None, train_x, train_y, valid_x, valid_y, test_x, test_y)}")
        held_out = (test_y, test_x)
        temporal_splits = []
        for i in range(0, 5):
            train_x, train_y, valid_x, valid_y, test_x, test_y = \
                get_split(splits.iloc[i], data, feature_columns, year_column, label_column)
            print(f"\t\tTemp split scores: {evaluate_model(None, train_x, train_y, valid_x, valid_y, test_x, test_y)}")
            temporal_splits.append((test_y, test_x))
        performance_evaluations.append((held_out, temporal_splits, 'SAPS II', 'orange'))

        # LR
        print(f"\nEvaluate LR model.")
        feature_columns, year_column, label_column, data = read_features_years_labels(args.data)
        feature_columns, data = add_categorical_dummies_and_nan_indicators(args.item_overview, feature_columns, data)
        assert len(feature_columns) == 1423 + 106 - 39 + 856  # Change 39 feat to dummies, 856 nan indicators
        removed_feature_columns = [f for f in feature_columns if f not in best_lr_features]
        data.drop(removed_feature_columns, axis=1, inplace=True)
        data = data[best_lr_features + [year_column] + [label_column]]
        feature_columns = best_lr_features

        assert len(feature_columns) == len(data.columns.tolist()) - 2
        assert len(feature_columns) == 130
        splits = create_splits(data, feature_columns, year_column, label_column)
        train_x, train_y, valid_x, valid_y, test_x, test_y = \
            get_split(splits.iloc[-1], data, feature_columns, year_column, label_column)
        # Z-normalize data for LR.
        train_x, valid_x, test_x = z_normalize_based_on_train(train_x, valid_x, test_x)
        clf = LogisticRegression(verbose=0)
        clf.set_params(**best_lr_params_130_features)
        clf.fit(train_x, train_y)
        print(f"\tFull split scores: {evaluate_model(clf, train_x, train_y, valid_x, valid_y, test_x, test_y)}")
        held_out = (test_y, clf.predict_proba(test_x)[:, 1])
        temporal_splits = []
        for i in range(0, 5):
            train_x, train_y, valid_x, valid_y, test_x, test_y = \
                get_split(splits.iloc[i], data, feature_columns, year_column, label_column)
            train_x, valid_x, test_x = z_normalize_based_on_train(train_x, valid_x, test_x)
            clf = LogisticRegression(verbose=0)
            assert len(feature_columns) == 130
            clf.set_params(**best_lr_params_130_features)
            clf.fit(train_x, train_y)
            print(f"\t\tTemp split scores: {evaluate_model(clf, train_x, train_y, valid_x, valid_y, test_x, test_y)}")
            temporal_splits.append((test_y, clf.predict_proba(test_x)[:, 1]))
        performance_evaluations.append((held_out, temporal_splits, 'LR', 'green'))

        # GBM
        print(f"\nEvaluate GBM model.")
        data, feature_columns, year_column, label_column, splits = create_gbm_data(args.data, args.item_overview)
        # Use full split for experiments.
        train_x, train_y, valid_x, valid_y, test_x, test_y = \
            get_split(splits.iloc[-1], data, feature_columns, year_column, label_column)
        clf = XGBClassifier(**{**best_gbm_params_full_features, 'n_jobs': 10})
        clf.fit(train_x, train_y)
        print(f"\tFull split scores: {evaluate_model(clf, train_x, train_y, valid_x, valid_y, test_x, test_y)}")
        held_out = (test_y, clf.predict_proba(test_x)[:, 1])
        temporal_splits = []
        for i in range(0, 5):
            train_x, train_y, valid_x, valid_y, test_x, test_y = \
                get_split(splits.iloc[i], data, feature_columns, year_column, label_column)
            clf = XGBClassifier(**{**best_gbm_params_full_features, 'n_jobs': 10})
            clf.fit(train_x, train_y)
            print(f"\t\tTemp split scores: {evaluate_model(clf, train_x, train_y, valid_x, valid_y, test_x, test_y)}")
            temporal_splits.append((test_y, clf.predict_proba(test_x)[:, 1]))
        performance_evaluations.append((held_out, temporal_splits, 'GBM', 'red'))

        # RNN
        print(f"\nEvaluate RNN model.")
        # Load specific rnn data that contains raw timeseries data.
        feature_columns, timeseries_features, train_ts_x, train_st_x, valid_ts_x, valid_st_x, test_ts_x, test_st_x,\
            train_y, valid_y, test_y = create_rnn_data(args.data_rnn, args.item_overview, split=-1)
        params = best_rnn_params_full_features
        # Set random seeds for reproducibility (only on same computer), see https://keras.io/getting-started/faq
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(params['seed'])
        python_random.seed(params['seed'])
        tf.random.set_seed(params['seed'])

        timeseries_in = keras.Input(shape=((21 * 24) + 1, len(timeseries_features)))
        if best_rnn_params_full_features['rnn_layers'] == 1:
            x = layers.LSTM(params['lstm_neurons'], activation='tanh',
                            recurrent_activation='sigmoid', dropout=params['dropout'],
                            recurrent_dropout=params['recurrent_dropout'])(timeseries_in)
        else:  # 2 rnn layers
            x = layers.LSTM(params['lstm_neurons'], activation='tanh',
                            recurrent_activation='sigmoid', dropout=params['dropout'],
                            recurrent_dropout=params['recurrent_dropout'], return_sequences=True)(timeseries_in)
            x = layers.LSTM(params['lstm_neurons'], activation='tanh', recurrent_activation='sigmoid',
                            dropout=params['dropout'], recurrent_dropout=params['recurrent_dropout'])(x)

        static_in = keras.Input(train_st_x.shape[1])
        x = layers.concatenate([x, static_in])
        x = layers.Dense(params['dense_neurons'], activation='relu')(x)
        x = layers.Dropout(params['dropout'])(x)
        output = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model([timeseries_in, static_in], output)
        model.summary()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=[keras.metrics.AUC(curve='PR'), keras.metrics.AUC(curve='ROC')])

        model.fit([train_ts_x, train_st_x], train_y, batch_size=params['batch_size'],
                  epochs=params['epochs'], validation_data=([valid_ts_x, valid_st_x], valid_y), verbose=0)
        eval_scores = evaluate_model(model, valid_x=[valid_ts_x, valid_st_x], valid_y=valid_y,
                                     test_x=[test_ts_x, test_st_x], test_y=test_y)
        print(f"\tFull split scores: {eval_scores}")
        held_out = (test_y, model.predict([test_ts_x, test_st_x]))
        temporal_splits = []
        for i in range(0, 5):
            feature_columns, timeseries_features, train_ts_x, train_st_x, valid_ts_x, valid_st_x, test_ts_x, test_st_x,\
                train_y, valid_y, test_y = create_rnn_data(args.data_rnn, args.item_overview, split=i)
            params = best_rnn_params_full_features
            # Set random seeds for reproducibility (only on same computer), see https://keras.io/getting-started/faq
            os.environ['PYTHONHASHSEED'] = '0'
            np.random.seed(params['seed'])
            python_random.seed(params['seed'])
            tf.random.set_seed(params['seed'])

            timeseries_in = keras.Input(shape=((21 * 24) + 1, len(timeseries_features)))
            if best_rnn_params_full_features['rnn_layers'] == 1:
                x = layers.LSTM(params['lstm_neurons'], activation='tanh',
                                recurrent_activation='sigmoid', dropout=params['dropout'],
                                recurrent_dropout=params['recurrent_dropout'])(timeseries_in)
            else:  # 2 rnn layers
                x = layers.LSTM(params['lstm_neurons'], activation='tanh',
                                recurrent_activation='sigmoid', dropout=params['dropout'],
                                recurrent_dropout=params['recurrent_dropout'], return_sequences=True)(timeseries_in)
                x = layers.LSTM(params['lstm_neurons'], activation='tanh', recurrent_activation='sigmoid',
                                dropout=params['dropout'], recurrent_dropout=params['recurrent_dropout'])(x)

            static_in = keras.Input(train_st_x.shape[1])
            x = layers.concatenate([x, static_in])
            x = layers.Dense(params['dense_neurons'], activation='relu')(x)
            x = layers.Dropout(params['dropout'])(x)
            output = layers.Dense(1, activation="sigmoid")(x)
            model = keras.Model([timeseries_in, static_in], output)
            model.summary()
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
                          loss=keras.losses.BinaryCrossentropy(),
                          metrics=[keras.metrics.AUC(curve='PR'), keras.metrics.AUC(curve='ROC')])
            model.fit([train_ts_x, train_st_x], train_y, batch_size=params['batch_size'],
                      epochs=params['epochs'], validation_data=([valid_ts_x, valid_st_x], valid_y), verbose=0)
            eval_scores = evaluate_model(model, valid_x=[valid_ts_x, valid_st_x], valid_y=valid_y,
                                         test_x=[test_ts_x, test_st_x], test_y=test_y)
            print(f"\t\tTemp split scores: {eval_scores}")
            temporal_splits.append((test_y, model.predict([test_ts_x, test_st_x])))

        performance_evaluations.append((held_out, temporal_splits, 'RNN', 'purple'))

        plot_performance(performance_evaluations, args.figures_output_path)

    if args.model == 'saps':
        pass

    elif args.model == 'gbm':
        data, feature_columns, year_column, label_column, splits = create_gbm_data(args.data, args.item_overview)
        # Use full split for experiments.
        train_x, train_y, valid_x, valid_y, test_x, test_y = \
            get_split(splits.iloc[-1], data, feature_columns, year_column, label_column)

        if args.parameter_tuning:
            print(f"Perform parameter tuning with {len(feature_columns)} features.")

            @output_wrapper
            def gbm_run(params):
                start = time.time()
                gbm_run_clf = XGBClassifier(**{**params, 'n_jobs': 1})
                gbm_run_clf.fit(train_x, train_y)

                eval_scores = evaluate_model(gbm_run_clf, train_x, train_y, valid_x, valid_y, None, None)
                # Return dictionary of all parameters and scores.
                result = {**params, **eval_scores}
                print('\t', datetime.now().strftime("%m/%d/%Y-%H:%M:%S"), f"({(time.time() - start):.2f}s):", result)
                return result

            global gbm_params
            print(f"Test {len(ParameterGrid(gbm_params))} parameter settings for gbm.")
            params_and_scores = \
                Parallel(n_jobs=8)(delayed(gbm_run)(params) for params in ParameterGrid(gbm_params))
            out = pd.DataFrame(params_and_scores)
            out.to_csv(f"/home/stefanhgm/MLUKM/Experiments/4_performance_comparison/"
                       f"{args.model}_parameter-tuning-{str(time.strftime('%Y%m%d-%H%M%S'))}.csv", sep='\t',
                       index=False)
        else:
            clf = XGBClassifier(**{**best_gbm_params_full_features, 'n_jobs': 10})
            clf.fit(train_x, train_y)
            scores = evaluate_model(clf, train_x, train_y, valid_x, valid_y, None, None)
            print(f"Result for best XBoost model: {scores}")

    elif args.model == 'rnn-features':
        # For rnn use raw timeseries data. Repeat similar process as for feature processing as in step 5.
        item_processing = read_item_processing_descriptions_from_excel(args.item_overview)
        db_conn = ResearchDBConnection(args.password)
        stays = db_conn.read_table_from_db('stays')
        stays_pid_case_ids = get_pid_case_ids_map(db_conn, stays)
        # Process stays similar to step 6.
        stays.loc[:, 'year_icu_out'] = pd.to_datetime(stays.loc[:, 'intbis'].dt.year, format="%Y")
        stays = stays.set_index('id', drop=False)
        stays = stays[['id', 'year_icu_out', 'label']]

        # Add 4 manually generated features as static inputs.
        old_index = stays.index.values.tolist()
        man1 = (static_is_3d_icu_readmission(args.password)[1][['stayid', 'numericvalue']]).set_index('stayid')
        man2 = (static_icu_station(args.password)[1][['stayid', 'textvalue']]).set_index('stayid')
        man3 = (static_icu_length_of_stay_days(args.password)[1][['stayid', 'numericvalue']]).set_index('stayid')
        man4 = (static_length_of_stay_before_icu_days(args.password)[1][['stayid', 'numericvalue']]).set_index('stayid')
        stays = stays.merge(man1, left_index=True, right_on='stayid', how='left')
        stays = stays.merge(man2, left_index=True, right_on='stayid', how='left')
        stays = stays.merge(man3, left_index=True, right_on='stayid', how='left')
        stays = stays.merge(man4, left_index=True, right_on='stayid', how='left')
        stays.rename(columns={stays.columns[-4]: 'Is 3-day ICU readmission',
                              stays.columns[-3]: 'ICU station',
                              stays.columns[-2]: 'ICU length of stay [days]',
                              stays.columns[-1]: 'Length of stay before ICU [days]'}, inplace=True)
        assert stays.shape[0] == 15589
        assert stays.index.values.tolist() == old_index
        # Important: considers stays always sorted by id to get the same ordering everywhere.
        stays = stays.sort_values(by='id')

        # Parse "included" items and merged items as specified by variable description.
        variables = item_processing.loc[
            ((item_processing['decision'] == 'included') | (item_processing['id'] >= 10000)) &
            (not args.variable_names or item_processing['variable_name'].str.contains('|'.join(args.variable_names))),
            ['id', 'variable_name', 'type', 'values', 'unit', 'feature_generation']]
        print(f"Detected {variables.shape[0]} variables to process.")

        # Process all specified features described in the variable description.
        feat_inputs = Parallel(n_jobs=30)(delayed(parallel_rnn_data_processing)(args.password, var, stays_pid_case_ids)
                                          for _, var in variables.iterrows())
        # To process in serial manner:
        # [parallel_feature_processing(args.password, var, stays_pid_case_ids) for _, var in variables.iterrows()]

        # Format data as for the usual feature export.
        feat_inputs = pd.DataFrame(feat_inputs).T
        feat_inputs = feat_inputs.rename(columns={col: variables.iloc[i]['variable_name'] for i, col in
                                                  enumerate(feat_inputs.columns.tolist())})
        # From step 6: Add year until stay lasts and label.
        stays.reset_index(drop=True, inplace=True)
        feat_inputs.reset_index(drop=True, inplace=True)
        output = pd.concat([feat_inputs, stays], axis=1)
        output = output[[c for c in output.columns.tolist() if c not in ['year_icu_out', 'label']] +
                        ['year_icu_out', 'label']]
        assert output.shape[0] == stays.shape[0]

        # Now sort stays again to the order of ids which is the same as for the usual output.
        output = output.sort_values(by='id')
        output.drop('id', axis=1, inplace=True)
        print(f"Write features and labels to {args.data_rnn}")
        if os.path.exists(args.data_rnn):
            os.remove(args.data_rnn)
        output.to_csv(args.data_rnn, sep=',', header=True, index=False, compression='gzip')

    elif args.model == 'rnn':
        # Load specific rnn data that contains raw timeseries data.
        feature_columns, timeseries_features, train_ts_x, train_st_x, valid_ts_x, valid_st_x, test_ts_x, test_st_x,\
            train_y, valid_y, test_y = create_rnn_data(args.data_rnn, args.item_overview, split=-1)

        params_and_scores = []
        if args.parameter_tuning:
            print(f"Perform parameter tuning with {len(feature_columns)} features.")
            global rnn_params
            for params in ParameterGrid(rnn_params):
                # Set random seeds for reproducibility (only on same computer), see https://keras.io/getting-started/faq
                os.environ['PYTHONHASHSEED'] = '0'
                np.random.seed(params['seed'])
                python_random.seed(params['seed'])
                tf.random.set_seed(params['seed'])

                print(f"Perform PT with {params}.")
                # Build LSTM model with additional static inputs.
                timeseries_in = keras.Input(shape=((21 * 24) + 1, len(timeseries_features)))
                if params['rnn_layers'] == 1:
                    x = layers.LSTM(params['lstm_neurons'], activation='tanh',
                                    recurrent_activation='sigmoid', dropout=params['dropout'],
                                    recurrent_dropout=params['recurrent_dropout'])(timeseries_in)
                else:  # 2 rnn layers
                    x = layers.LSTM(params['lstm_neurons'], activation='tanh',
                                    recurrent_activation='sigmoid', dropout=params['dropout'],
                                    recurrent_dropout=params['recurrent_dropout'], return_sequences=True)(timeseries_in)
                    x = layers.LSTM(params['lstm_neurons'], activation='tanh', recurrent_activation='sigmoid',
                                    dropout=params['dropout'], recurrent_dropout=params['recurrent_dropout'])(x)
                static_in = keras.Input(train_st_x.shape[1])
                x = layers.concatenate([x, static_in])
                x = layers.Dense(params['dense_neurons'], activation='relu')(x)
                x = layers.Dropout(params['dropout'])(x)
                output = layers.Dense(1, activation="sigmoid")(x)
                model = keras.Model([timeseries_in, static_in], output)
                model.summary()
                model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
                              loss=keras.losses.BinaryCrossentropy(),
                              metrics=[keras.metrics.AUC(curve='PR'), keras.metrics.AUC(curve='ROC')])

                # Custom callback for prediction on validation set.
                class PredictionCallback(keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs={}):
                        epoch_score = evaluate_model(model, valid_x=[valid_ts_x, valid_st_x], valid_y=valid_y,
                                                     test_x=None, test_y=None)
                        # Store results of each epoch.
                        params_and_scores.append({**params, 'epoch': epoch+1, **epoch_score})
                        print(f"Epoch {epoch+1} prediction: {epoch_score}")
                try:
                    model.fit([train_ts_x, train_st_x], train_y, batch_size=params['batch_size'],
                              epochs=params['epochs'], validation_data=([valid_ts_x, valid_st_x], valid_y),
                              verbose=0, callbacks=[PredictionCallback()])
                except InvalidArgumentError as e:
                    print(f"\tInvalidArgument exception. Skipped.")
                    print(e)
                except StopIteration as e:
                    print(f"\tStopIteration exception. Skipped.")
                    print(e)

            out = pd.DataFrame(params_and_scores)
            out.to_csv(f"/home/stefanhgm/MLUKM/Experiments/4_performance_comparison/"
                       f"{args.model}_parameter-tuning-{str(time.strftime('%Y%m%d-%H%M%S'))}.csv", sep='\t',
                       index=False)


def create_rnn_data(data_rnn, item_overview, split):
    feature_columns, year_column, label_column, data = read_features_years_labels(data_rnn)
    assert len(feature_columns) == 169 + 4

    # Read timeseries data as float. NaNs in timeseries data already replaced when creating rnn features.
    timeseries_features =\
        [f for f in feature_columns if (is_string_dtype(data[f]) and data.loc[0, f].startswith('['))]
    data[timeseries_features] = data[timeseries_features].applymap(lambda string_list: json.loads(string_list))

    static_features = [f for f in feature_columns if f not in timeseries_features]
    assert len(timeseries_features) + len(static_features) == len(feature_columns)
    assert len(static_features) == 21
    assert len(timeseries_features) == 152

    # Add dummies and nan indicators for static features.
    static_features, data =\
        add_categorical_dummies_and_nan_indicators(item_overview, static_features, data, only_variables=True)

    feature_columns = timeseries_features + static_features
    splits = create_splits(data, feature_columns, year_column, label_column)
    train_x, train_y, valid_x, valid_y, test_x, test_y = \
        get_split(splits.iloc[split], data, feature_columns, year_column, label_column)
    train_ts_x = train_x[timeseries_features]
    train_ts_x =\
        np.transpose(np.array([np.array(train_ts_x[i].tolist()) for i in train_ts_x.columns.tolist()]), (1, 2, 0))
    train_st_x = train_x[static_features]
    valid_ts_x = valid_x[timeseries_features]
    valid_ts_x =\
        np.transpose(np.array([np.array(valid_ts_x[i].tolist()) for i in valid_ts_x.columns.tolist()]), (1, 2, 0))
    valid_st_x = valid_x[static_features]
    test_ts_x = test_x[timeseries_features]
    test_ts_x =\
        np.transpose(np.array([np.array(test_ts_x[i].tolist()) for i in test_ts_x.columns.tolist()]), (1, 2, 0))
    test_st_x = test_x[static_features]

    # Z-normalize data based on training data.
    ts_mean = np.mean(train_ts_x, axis=(0, 1))
    ts_std = np.std(train_ts_x, axis=(0, 1))
    st_mean = np.mean(train_st_x, axis=0)
    st_std = np.std(train_st_x, axis=0)
    train_ts_x = (train_ts_x - ts_mean) / ts_std
    train_st_x = (train_st_x - st_mean) / st_std
    valid_ts_x = (valid_ts_x - ts_mean) / ts_std
    valid_st_x = (valid_st_x - st_mean) / st_std
    test_ts_x = (test_ts_x - ts_mean) / ts_std
    test_st_x = (test_st_x - st_mean) / st_std

    return feature_columns, timeseries_features, train_ts_x, train_st_x, valid_ts_x, valid_st_x, test_ts_x, test_st_x,\
        train_y, valid_y, test_y


@output_wrapper
def parallel_rnn_data_processing(password, variable, stays_pid_case_ids):
    db_conn = ResearchDBConnection(password)
    stays = db_conn.read_table_from_db('stays')
    var_id, var_name, var_type, var_unit, feat_generation =\
        variable[['id', 'variable_name', 'type', 'unit', 'feature_generation']]
    feat_generation = [] if not feat_generation else json.loads(feat_generation)
    is_static = 'static' in feat_generation or 'static_per-hosp-stay' in feat_generation or\
        'static_per-icu-stay' in feat_generation
    is_timeseries = 'timeseries_low' in feat_generation or 'timeseries_medium' in feat_generation or\
        'timeseries_high' in feat_generation
    is_flow = 'flow' in feat_generation or var_name == 'Time on automatic ventilation'
    is_medication = 'medication' in feat_generation
    is_intervention = 'intervention_per-icu-stay' in feat_generation
    assert any([is_static, is_timeseries, is_flow, is_medication, is_intervention])

    # Read values of included patients for a variable.
    all_cases_of_all_patients = [item for sublist in list(stays_pid_case_ids.values()) for item in sublist]
    values = db_conn.read_values_from_db(var_name, var_type, case_id=all_cases_of_all_patients)
    # Derive datatype boolean a (treated as numerical) if only "true", "false" as strings.
    non_null = ~(values['data'].isnull())
    try:
        if is_string_dtype(values.loc[non_null, 'data']) and\
                all(values.loc[non_null, 'data'].str.lower().isin(['true', 'false'])):
            values.loc[non_null, 'data'] = values.loc[non_null, 'data'].str.lower().replace({'true': 1., 'false': 0.})
            var_type = 'continuous'
    except AttributeError:
        # In case no all values are strings.
        pass

    # Resulting inputs for all features.
    feat_inputs = []
    print(f"\tProcess variable {var_name} (id: {var_id}, type: {var_type}).")

    # Important: considers stays always sorted by id to get the same ordering everywhere.
    stays = stays.sort_values(by='id')
    for idx, _ in stays.iterrows():
        icu_start = stays.loc[idx, 'intvon']
        icu_end = stays.loc[idx, 'intbis']
        td_icu_stay_start = icu_end - icu_start
        td_hosp_stay_start = icu_end - stays.loc[idx, 'ukmvon']
        patient_cases = stays_pid_case_ids[stays.loc[idx, 'patientid']]
        patient_values_mask = (values['patientid'].isin(patient_cases))

        # Set time span according to feature method.
        if 'static' in feat_generation:
            time_span = decode_feature_time_span('per-patient', td_icu_stay_start, pd.Timedelta("100000 days"))
        elif 'static_per-hosp-stay' in feat_generation:
            time_span = decode_feature_time_span('per-hosp-stay', td_icu_stay_start, td_hosp_stay_start)
        elif 'static_per-icu-stay' in feat_generation:
            time_span = decode_feature_time_span('per-icu-stay', td_icu_stay_start, td_icu_stay_start)
        else:
            # Timeseries data for LSTM limited to hospital stay.
            time_span = decode_feature_time_span('per-hosp-stay', td_icu_stay_start, td_hosp_stay_start)

        # Determine all patient values for a given time span.
        patient_values_in_time_span = \
            values.loc[patient_values_mask & (values['displaytime'] >= icu_end - time_span) &
                       (values['displaytime'] <= icu_end), ['data', 'displaytime']].copy()
        # Remove nan values, no feature function needs them.
        patient_values_in_time_span.dropna(inplace=True)

        # Distinguish static features and timeseries features.
        if is_static:
            patient_values_in_time_span = patient_values_in_time_span.sort_values(by='displaytime')
            feat = patient_values_in_time_span.iloc[-1]['data'] if patient_values_in_time_span.shape[0] > 0 else np.nan
            feat_inputs.append(feat)
        else:
            earliest_entry = patient_values_in_time_span['displaytime'].min()
            # Reduce data to at most 21d before icu discharge.
            patient_values_in_time_span = patient_values_in_time_span.loc[
                patient_values_in_time_span['displaytime'] >= icu_end - pd.Timedelta("21 days")]
            patient_values_in_time_span = patient_values_in_time_span.set_index('displaytime')
            # Set first/last value to nan if they don't exist to have proper boundaries for resample.
            if (icu_end - pd.Timedelta("21 days")) not in patient_values_in_time_span.index:
                if (not is_timeseries) or (var_name not in timeseries_imputation.keys()):
                    patient_values_in_time_span.loc[icu_end - pd.Timedelta("21 days")] = np.nan
                else:
                    # If imputation value exists use it at the start to fill missing values.
                    # If no value exists at all the imputation will be carried through till the end.
                    patient_values_in_time_span.loc[icu_end - pd.Timedelta("21 days")] = timeseries_imputation[var_name]
            if icu_end not in patient_values_in_time_span.index:
                if not is_timeseries:
                    patient_values_in_time_span.loc[icu_end] = np.nan
                else:
                    # Copy last value to the end to make it compatible with forward filling.
                    patient_values_in_time_span = patient_values_in_time_span.sort_index()
                    patient_values_in_time_span.loc[icu_end] = patient_values_in_time_span.iloc[-1]
            # Resample to 1h intervals which corresponds to 4 times the smallest intervals used for ICU recordings.
            if is_timeseries:
                patient_values_in_time_span =\
                    patient_values_in_time_span[~patient_values_in_time_span.index.duplicated(keep='first')]
                # First resample to smallest interval with forward fill to prevent nan values.
                patient_values_in_time_span = \
                    patient_values_in_time_span.resample('15min', origin='start').ffill()\
                    .resample('1h', origin='start').median()
            elif is_flow:
                patient_values_in_time_span = \
                    patient_values_in_time_span.resample('1h', origin='start', closed='right').sum()
            elif is_medication:
                patient_values_in_time_span = \
                    patient_values_in_time_span.resample('1h', origin='start', closed='right').count()
            elif is_intervention:
                patient_values_in_time_span =\
                    patient_values_in_time_span.resample('1h', origin='start', closed='right').max()

            # Set all values before first value to nan.
            if is_timeseries:
                patient_values_in_time_span.loc[
                    patient_values_in_time_span.index < earliest_entry, 'data'] = np.nan
                assert patient_values_in_time_span.index.min() == icu_end - pd.Timedelta("21 days")
                assert patient_values_in_time_span.index.max() == icu_end
            else:
                patient_values_in_time_span.loc[
                    patient_values_in_time_span.index < (earliest_entry - pd.Timedelta("1 hours")), 'data'] = np.nan
                assert patient_values_in_time_span.index.min() == \
                    icu_end - pd.Timedelta("21 days") - pd.Timedelta("1 hours")
                assert patient_values_in_time_span.index.max() == icu_end - pd.Timedelta("1 hours")
            assert patient_values_in_time_span.shape[0] == (21 * 24) + 1
            feat_inputs.append(patient_values_in_time_span['data'].to_list())

    # For non-static replace nan with minimum - 1. Have to do it at the end bc only here minimum known.
    if not is_static:
        minimum = min([min([x for x in row if not pd.isna(x)], default=9999) for row in feat_inputs], default=[9999])
        assert minimum != 9999
        for i in range(0, len(feat_inputs)):
            feat_inputs[i] = [x if not pd.isna(x) else (minimum - 1) for x in feat_inputs[i]]

    assert len(feat_inputs) == 15589
    return feat_inputs


def create_gbm_data(data_path, item_overview_path):
    # Load data and replace categorical values as for LR models. Not nan indicators and z-normalization though.
    feature_columns, year_column, label_column, data = read_features_years_labels(data_path)
    assert len(feature_columns) == 1423 and len(feature_columns) == len(set(feature_columns))
    # Remove "[", "]" and "<" from column names for XGBoost.
    regex = re.compile(r'[\[\]<]', re.IGNORECASE)
    feature_columns = \
        [regex.sub("_", col) if any(x in str(col) for x in {'[', ']', '<'}) else col for col in feature_columns]
    data.columns = feature_columns + [year_column] + [label_column]
    # Add dummies for categorical variables just as for LR model.
    feature_columns, data = add_categorical_dummies_and_nan_indicators(item_overview_path, feature_columns, data,
                                                                       add_nan_indicators=False)
    assert len(feature_columns) == 1423 + 106 - 39  # Change 39 feat to dummies
    splits = create_splits(data, feature_columns, year_column, label_column)
    return data, feature_columns, year_column, label_column, splits


if __name__ == '__main__':
    main()
