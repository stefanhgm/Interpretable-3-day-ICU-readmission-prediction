import argparse
import random

from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from experiments.exp1_LR_EBM_parameter_tuning_all_features import evaluate_model, \
    best_ebm_params_2d_model, plot_performance
from experiments.exp4_performance_comparison import best_gbm_params_full_features
from experiments.variables import seed
from helper.util import read_features_years_labels
from experiments.custom_ebm.ebm_enable_unk_min_samples_bin import ExplainableBoostingClassifier

# Subset of feature for MIMIC EBM model.
feature_columns_ebm = [
    'Age (static all data)', 'Antithrombotic agents prophylactic dosage (days since last application per icu stay)',
    'BE (iqr 1d)', 'BE (median 12h)', 'BE (min 12h)', 'BE (trend per day 3d)', 'Bilirubin total (max 7d)',
    'Blood Urea Nitrogen (min 3d)', 'Blood volume out (extrapolate 3d)', 'Blood volume out (extrapolate 7d)',
    'Body core temperature (median 1d)', 'Body core temperature (trend per day 1d)', 'C-reactive protein (max 3d)',
    'CK (min 7d)', 'CK-MB (max 3d)', 'CK-MB (median 3d)', 'Calcium (max 1d)', 'Chloride (min 1d)',
    'Chloride (trend per day 3d)', 'Diastolic blood pressure (median 1d)', 'Drugs for constipation (has entry 1d)',
    'Estimated respiratory rate (median 1d)', 'GCS score (min 3d)', 'Gamma-GT (median 7d)', 'Glucose (median 3d)',
    'Heart rate (iqr 4h)', 'Heart rate (min 1d)', 'Heart rate (min 4h)', 'Hematocrit (max 3d)',
    'Hematocrit (median 12h)', 'Hemoglobin (max 3d)',
    'Is on automatic ventilation (days since last application per icu stay)', 'Lactate (max 3d)', 'Lactate (min 12h)',
    'Length of stay before ICU (static all data)', 'Leucocytes (iqr 3d)', 'Leucocytes (median 1d)',
    'Mean blood pressure (median 12h)', 'Mean blood pressure (median 4h)', 'MetHb (min 12h)', 'O2 saturation (min 12h)',
    'Phosphate (max 1d)', 'Phosphate (min 7d)', 'Potassium (median 1d)', 'RAS scale (max 1d)', 'RAS scale (max 3d)',
    'Sodium (median 3d)', 'Sodium (trend per day 3d)', 'Systolic blood pressure (iqr 12h)',
    'Thrombocytes (trend per day 7d)', 'Tubus exists (days since last application per icu stay)',
    'Urine volume out (extrapolate 1d)', 'Urine volume out (extrapolate 7d)', 'eGFR (trend per day 7d)',
    'pCO2 (iqr 1d)', 'pCO2 (median 1d)', 'pCO2 (min 3d)', 'pH (iqr 1d)', 'pH (median 1d)', 'pH (median 3d)',
    'pH (trend per day 3d)', 'pO2 (iqr 12h)', 'pO2 (min 12h)', 'paO2/FiO2 (median 1d)', 'paO2/FiO2 (median 3d)',
    'paO2/FiO2 (trend per day 3d)'
]


def main():
    parser = argparse.ArgumentParser(description='5. experiment: external validation of EBM model on MIMIC-IV.')
    parser.add_argument('password', type=str, help='Database password.')
    parser.add_argument('--data', type=str, help='CSV file containing features and targets.')
    parser.add_argument('--figures_output_path', type=str, help='Directory where generated figures will be stored.')
    args, _ = parser.parse_known_args()

    feature_columns, year_column, label_column, data = read_features_years_labels(args.data)
    assert len(feature_columns) == 515
    assert len(feature_columns_ebm) == 66

    # Original cohort has 891 positive, 14698 negative (total 15589) cases, i.e. pos. label rate of 0.05715568670216178.
    # MIMIC: 1626 positive, 17482 negative (total 19108), i.e. pos. label rate of 0.08509524806363827.
    # Adjust label rate of mimic data to make precision comparable.
    positive = data.loc[data['label'] == 1].shape[0]
    negative = data.loc[data['label'] == 0].shape[0]
    print(f"Original MIMIC cohort: {positive} positive, {negative} negative, {positive / (positive + negative)} ratio.")
    negative_indices = data.loc[data['label'] == 0].index.tolist()
    # Make choices reproducible.
    random.seed(seed)
    resampled = random.choices(negative_indices, k=int((1666/0.05715568670216178) - 17442 - 1666))
    data = pd.concat([data, data.loc[resampled]], axis=0, ignore_index=True)
    # Shuffle data.
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    positive = data.loc[data['label'] == 1].shape[0]
    negative = data.loc[data['label'] == 0].shape[0]
    print(f"Resampling negative: {positive} positive, {negative} negative, {positive / (positive + negative)} ratio.")

    # Use anchor years 2008 - 2016 for training and split up 2017-2019
    train = data.loc[(data[year_column] <= datetime(2016, 1, 1))]
    valid_test = data.loc[(data[year_column] > datetime(2016, 1, 1))]
    assert train.shape[0] + valid_test.shape[0] == data.shape[0]
    train_x_ebm = train[feature_columns_ebm].copy()
    train_x_gbm = train[feature_columns].copy()
    train_y = train[label_column].copy()
    x_valid_test_ebm = valid_test[feature_columns_ebm].copy()
    x_valid_test_gbm = valid_test[feature_columns].copy()
    y_valid_test = valid_test[label_column].copy()
    # If additional validation set necessary.
    # valid_x, test_x, valid_y, test_y = \
    #     train_test_split(x_valid_test, y_valid_test, test_size=0.5, random_state=seed)
    # assert valid_x.shape[0] + test_x.shape[0] == x_valid_test.shape[0]

    # EBM
    clf = ExplainableBoostingClassifier()
    clf.set_params(**{**best_ebm_params_2d_model, 'interactions': 0})
    clf.fit(train_x_ebm, train_y)
    # Store model.
    # store_ebm_model(clf, get_timestamp(), args.password)
    print(f"\tFull scores EBM: {evaluate_model(clf, train_x_ebm, train_y, None, None, x_valid_test_ebm, y_valid_test)}")
    held_out_ebm = (y_valid_test, clf.predict_proba(x_valid_test_ebm)[:, 1])

    # GBM
    clf = XGBClassifier(**{**best_gbm_params_full_features, 'n_jobs': 10})
    clf.fit(train_x_gbm, train_y)
    print(f"\tFull scores GBM: {evaluate_model(clf, train_x_gbm, train_y, None, None, x_valid_test_gbm, y_valid_test)}")
    held_out_gbm = (y_valid_test, clf.predict_proba(x_valid_test_gbm)[:, 1])

    performance_evaluations = []
    temporal_splits_ebm = []
    temporal_splits_gbm = []
    for i in range(0, 5):
        temporal_split = data.loc[(data[year_column] <= datetime(2016, 1, 1))]
        temporal_split_x_ebm = temporal_split[feature_columns_ebm].copy()
        temporal_split_x_gbm = temporal_split[feature_columns].copy()
        temporal_split_y = temporal_split[label_column].copy()
        train_x_ebm, test_x_ebm, train_x_gbm, test_x_gbm, train_y, test_y = \
            train_test_split(temporal_split_x_ebm, temporal_split_x_gbm, temporal_split_y, test_size=0.15,
                             random_state=seed + i)

        # EBM
        clf = ExplainableBoostingClassifier()
        clf.set_params(**{**best_ebm_params_2d_model, 'interactions': 0})
        clf.fit(train_x_ebm, train_y)
        print(f"\t\tTemp split scores EBM: {evaluate_model(clf, train_x_ebm, train_y, None, None, test_x_ebm, test_y)}")
        temporal_splits_ebm.append((test_y, clf.predict_proba(test_x_ebm)[:, 1]))

        # GBM
        clf = XGBClassifier(**{**best_gbm_params_full_features, 'n_jobs': 10})
        clf.fit(train_x_gbm, train_y)
        print(f"\t\tTemp split scores GBM: {evaluate_model(clf, train_x_gbm, train_y, None, None, test_x_gbm, test_y)}")
        temporal_splits_gbm.append((test_y, clf.predict_proba(test_x_gbm)[:, 1]))

    performance_evaluations.append((held_out_ebm, temporal_splits_ebm, 'EBM MIMIC', 'blue'))
    performance_evaluations.append((held_out_gbm, temporal_splits_gbm, 'GBM MIMIC', 'red'))

    plot_performance(performance_evaluations, args.figures_output_path)


if __name__ == '__main__':
    main()
