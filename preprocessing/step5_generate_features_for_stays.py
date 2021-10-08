"""
Script to convert the variables and values into features and inputs for the stays.
Also adds imputation for selected timeseries variables.
"""
import argparse
import datetime
import json

from joblib import Parallel, delayed
from scipy.stats import iqr

from helper.io import read_item_processing_descriptions_from_excel
from helper.util import output_wrapper, parse_time, format_timedelta, get_pid_case_ids_map
from research_database.research_database_communication import ResearchDBConnection, resolve_data_column_name
from pandas.api.types import is_string_dtype

import pandas as pd
import numpy as np

# Feature methods and the according feature generation methods.
feat_methods = {
    # Non-SAPS features.
    # 'static': {}, 'static_per-hosp-stay': {},
    # 'timeseries_high': {}, 'timeseries_medium': {}, 'timeseries_low': {},
    # 'timeseries_bool_high': {}, 'timeseries_bool_medium': {}, 'timeseries_bool_low': {},
    'static': {'per-patient': ['last']},
    'static_per-hosp-stay': {'per-hosp-stay': ['last']},
    'static_per-icu-stay': {'per-icu-stay': ['last']},
    'intervention_per-icu-stay': {'per-icu-stay': ['has-true', 'true-until-discharge']},
    'timeseries_high': {'4h': ['median', 'iqr', 'min', 'max', 'trend'],
                        '12h': ['median', 'iqr', 'min', 'max', 'trend'],
                        '1d': ['median', 'iqr', 'min', 'max', 'trend']},
    'timeseries_medium': {'12h': ['median', 'iqr', 'min', 'max', 'trend'],
                          '1d': ['median', 'iqr', 'min', 'max', 'trend'],
                          '3d': ['median', 'iqr', 'min', 'max', 'trend']},
    'timeseries_low': {'1d': ['median', 'iqr', 'min', 'max', 'trend'],
                       '3d': ['median', 'iqr', 'min', 'max', 'trend'],
                       '7d': ['median', 'iqr', 'min', 'max', 'trend']},
    'flow': {'1d': ['extrapolate'],
             '3d': ['extrapolate'],
             '7d': ['extrapolate']},
    'medication': {'1d': ['has-entry', 'unique'],
                   '3d': ['has-entry', 'unique'],
                   '7d': ['has-entry', 'unique']},

    # Additional features.
    'sum_per-icu-stay': {'per-icu-stay': ['sum']},  # Right now only for duration on automatic ventilation.

    # Special SAPS features for SAPS Score model.
    'saps_static': {}, 'saps_static_per-hosp-stay': {}, 'saps_static_24h': {}, 'saps_static_bool': {},
    'saps_min_max_24h': {}, 'saps_max_24h': {}, 'saps_min_24h': {}, 'saps_extrapolate_24h': {}
    # 'saps_last': ['saps_last'],
    # 'saps_last_per-hosp-stay': ['saps_last_per-hosp-stay'],
    # 'saps_last_24h': ['saps_last_1d0h0m'],
    # 'saps_last_bool': ['saps_last_bool'],
    # 'saps_min_max_24h': ['saps_min_1d0h0m', 'saps_max_1d0h0m'],
    # 'saps_max_24h': ['saps_max_1d0h0m'],
    # 'saps_min_24h': ['saps_min_1d0h0m'],
    # 'saps_extrapolate_24h': ['saps_extrapolate_1d0h0m']
}

# Defines basic feature functions.
feat_func_map = {
    'min': np.min,
    'max': np.max,
    'median': np.median,
    'mean': np.mean,
    'sd': np.std,
    'sum': np.sum,
    'iqr': iqr,
    'unique': lambda x: np.unique(x).shape[0],
    'last': lambda x: x[len(x) - 1],
    'has-entry': lambda x: x.shape[0] > 0,
    'has-true': lambda x: 1. if (x.shape[0] > 0 and True in x) else 0.,
}

# Defines feature functions with additional time-span argument.
feat_func_interval_map = {
    'mean-dosage': lambda x, y: np.sum(x) / (y / pd.Timedelta("1 days"))
}

# Defines feature functions with additional time-span, icu stay start, icu stay end arguments.
feat_func_interval_start_end_map = {
    'extrapolate': lambda x, y, stay_start, stay_end: feat_func_extrapolate(x, y, stay_start, stay_end)
}

# Defines feature functions with additional times argument.
feat_func_with_times_map = {
    'true-until-discharge': lambda x, y: np.nan if (x.shape[0] == 0 or True not in x) else
                                                   (-y[(np.where(x == True))[0][-1]] / np.timedelta64(1, 'D'))
}

# Defines feature functions with additional time-span argument.
feat_func_with_times_span_map = {
    # x: time in days, y: values
    # Define trend of zero if only single recording, to get undefined only for empty intervals.
    'trend': lambda x, y: np.nan if (x.shape[0] == 0 or y.shape[0] == 0) else
                                    (0 if np.unique(x).shape[0] == 1 or np.unique(y).shape[0] == 1 else
                                     np.polyfit(x, y, deg=1)[0])
}

# For selected variable define imputation with default values.
timeseries_imputation = {
    # Used default values of scores.
    'GCS Motor': 6,
    'GCS Verbal': 5,
    'GCS Eye': 4,
    'GCS score': 15,
    'RAS scale': 0,
    # Used median values in the data.
    'pO2': 88.8,
    'pCO2': 38.9,
    'pH': 7.417,
    'BE': 0.3,
    'Bicarbonate': 24.5,
    'Leucocytes': 11.49,
    'Thrombocytes': 166.0,
    'Creatinine': 0.9,
    'GPT (ALT)': 26.0,
    'CK': 302.0,
    'Body core temperature': 37.3,
    'Systolic blood pressure': 114,
    'Diastolic blood pressure': 57,
    'Mean blood pressure': 75,
    'O2 saturation': 97.0,
    'Chloride': 109.0,
    'COHb': 1.4,
    'MetHb': 0.9,
    'Hemoglobin': 9.6,
    'Hematocrit': 29.8,
    'Potassium': 4.3,
    'Sodium': 138.0,
    'Calcium': 1.16,
    'Glucose': 134.0,
    'Lactate': 1.1,
    'Quick': 75.0,
    'PTT': 42.0,
    'eGFR': 77.60777,
    'paO2/FiO2': 246.,
}


def main():
    parser = argparse.ArgumentParser(description='Extract features.')
    parser.add_argument('password', type=str, help='Database password.')
    parser.add_argument('item_overview', type=str, help='Description of all PDMS items and generated variables.')
    parser.add_argument('--variable_names', nargs='+', help='Only convert specified variables.', default='')
    args, _ = parser.parse_known_args()

    item_processing = read_item_processing_descriptions_from_excel(args.item_overview)
    db_conn = ResearchDBConnection(args.password)
    stays = db_conn.read_table_from_db('stays')
    stays_pid_case_ids = get_pid_case_ids_map(db_conn, stays)

    # Parse "included" items and merged items as specified by variable description.
    variables = item_processing.loc[
        ((item_processing['decision'] == 'included') | (item_processing['id'] >= 10000)) &
        (not args.variable_names or item_processing['variable_name'].str.contains('|'.join(args.variable_names))),
        ['id', 'variable_name', 'type', 'values', 'unit', 'feature_generation']]

    # Before processing feature generations per variable, check that they are all formatted correctly.
    num_feature_generations = 0
    num_features = 0
    for idx, description in variables.iterrows():
        generation = [] if not description['feature_generation'] else json.loads(description['feature_generation'])
        # If timeseries feature and categorical (bool) replace with appropriate generation method.
        if description['type'] == 'categorical':
            for g in generation:
                if 'timeseries' in g:
                    raise Exception("Timeseries features not defined for categorical variables.")
        for g in generation:
            num_feature_generations += 1
            if g not in feat_methods.keys():
                raise Exception(f"Found undefined feature method {g} for variable {description['variable_name']}.")
            else:
                num_features += sum([len(feat_methods[g][time]) for time in feat_methods[g].keys()])
        variables.at[idx, 'feature_generation'] = generation
    print(f"Detected {variables.shape[0]} variables, {num_feature_generations} feature generations, and {num_features} "
          f"features to process.")

    # Process all specified features described in the variable description.
    Parallel(n_jobs=40)(delayed(parallel_feature_processing)(args.password, var, stays_pid_case_ids)
                        for _, var in variables.iterrows())
    # To process in serial manner:
    # [parallel_feature_processing(args.password, var, stays_pid_case_ids) for _, var in variables.iterrows()]


@output_wrapper
def parallel_feature_processing(password, variable, stays_pid_case_ids):
    db_conn = ResearchDBConnection(password)
    stays = db_conn.read_table_from_db('stays')
    var_id, var_name, var_type, var_unit, feat_generation =\
        variable[['id', 'variable_name', 'type', 'unit', 'feature_generation']]
    feature_method_dicts = [feat_methods[g] for g in feat_generation]

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
    feat_inputs = pd.DataFrame(columns=['stayid', 'featurename', resolve_data_column_name(var_type)])
    print(f"\tProcess variable {var_name} (id: {var_id}, type: {var_type}) with feature generations {feat_generation}.")

    # For each stay consider all recordings of patient before stay end and apply all feature methods.
    feature_names = set()
    for idx, _ in stays.iterrows():
        icu_start = stays.loc[idx, 'intvon']
        icu_end = stays.loc[idx, 'intbis']
        td_icu_stay_start = icu_end - icu_start
        td_hosp_stay_start = icu_end - stays.loc[idx, 'ukmvon']
        patient_cases = stays_pid_case_ids[stays.loc[idx, 'patientid']]
        patient_values_mask = (values['patientid'].isin(patient_cases))

        # For each stay process all feature generation dictionaries.
        for feature_method_dict in feature_method_dicts:

            # Process all features of the same time span together.
            for time_entry in feature_method_dict.keys():
                time_span = decode_feature_time_span(time_entry, td_icu_stay_start, td_hosp_stay_start)
                # Determine all patient values for a given time span.
                patient_values_in_time_span =\
                    values.loc[patient_values_mask & (values['displaytime'] >= icu_end - time_span) &
                               (values['displaytime'] <= icu_end)].copy()
                # Remove nan values, no feature function needs them.
                patient_values_in_time_span.dropna(inplace=True)

                # Impute selected timeseries variables.
                if patient_values_in_time_span['data'].shape[0] == 0 and var_name in timeseries_imputation.keys():
                    patient_values_in_time_span = patient_values_in_time_span.append(
                        {'patientid': stays.loc[idx, 'fallnummer'], 'displaytime': icu_end,
                         'data': timeseries_imputation[var_name]}, ignore_index=True)

                if patient_values_in_time_span['data'].shape[0] > 0:
                    # Set time relative to stay end.
                    patient_values_in_time_span.loc[:, 'time_to_stay_end'] = \
                        patient_values_in_time_span.loc[:, 'displaytime'] - icu_end
                    assert all(patient_values_in_time_span['time_to_stay_end'] >= -time_span)
                    assert all(patient_values_in_time_span['time_to_stay_end'] <= datetime.timedelta(seconds=0))

                # Determine concrete feature value.
                for feat_method in feature_method_dict[time_entry]:
                    # Determine feature function and name.
                    feat_func_name, feat_name =\
                        decode_func_and_feat_name(var_id, var_name, var_unit, time_entry, feat_method)
                    feature_names.add(feat_name)

                    feat_value = np.nan
                    if patient_values_in_time_span['data'].shape[0] > 0:

                        # Determine the function for the given function name.
                        if feat_func_name in feat_func_map.keys():
                            function = feat_func_map[feat_func_name]
                        elif feat_func_name in feat_func_interval_map.keys():
                            function = lambda x: feat_func_interval_map[feat_func_name](x, time_span)
                        elif feat_func_name in feat_func_interval_start_end_map.keys():
                            function = lambda x: feat_func_interval_start_end_map[feat_func_name](x, time_span,
                                                                                                  icu_start, icu_end)
                        elif feat_func_name in feat_func_with_times_map.keys():
                            function = lambda x: feat_func_with_times_map[feat_func_name] \
                                (x, (patient_values_in_time_span.loc[:, 'time_to_stay_end']).to_numpy())
                        elif feat_func_name in feat_func_with_times_span_map.keys():
                            # On the time axis use number of days to get trend per day,
                            function = lambda x: feat_func_with_times_span_map[feat_func_name] \
                                ((patient_values_in_time_span.loc[:, 'time_to_stay_end']
                                  / datetime.timedelta(days=1)).to_numpy(), x)
                        else:
                            raise Exception(f"No valid feature function found (feature function: {feat_func_name}).")

                        feat_value = function(patient_values_in_time_span['data'].to_numpy())

                    feat_inputs = feat_inputs.append({'stayid': stays.loc[idx, 'id'], 'featurename': feat_name,
                                                      feat_inputs.columns[2]: feat_value}, ignore_index=True)

    print(f"\t\tProcessed features: {feature_names}.")
    db_conn.write_features_and_inputs_to_db(list(feature_names), feat_inputs)


def decode_feature_time_span(time_entry, td_icu_stay_start, td_hosp_stay_start):
    if time_entry == 'per-patient':
        time_span = pd.Timedelta("100000 days")
    elif time_entry == 'per-icu-stay':
        time_span = td_icu_stay_start
    elif time_entry == 'per-hosp-stay':
        time_span = td_hosp_stay_start
    else:
        time_span = parse_time(time_entry)

    return time_span


def decode_func_and_feat_name(var_id, var_name, var_unit, time_entry, function_name):
    # Deal with special prefixes.
    prefix = ''
    if function_name.startswith('saps_'):
        function_name = function_name[5:]
        prefix = 'SAPS II '

    # Rename function ame for better understanding.
    display_function_name = function_name
    if display_function_name == 'last':
        display_function_name = 'static'
    elif display_function_name == 'has-entry':
        display_function_name = 'has entry'
    elif display_function_name == 'true-until-discharge':
        display_function_name = 'days since last application'
    elif display_function_name == 'trend':
        display_function_name = 'trend per day'

    if time_entry == 'per-patient':
        time_span_name = 'all data'
    elif time_entry == 'per-icu-stay':
        time_span_name = 'per icu stay'
    elif time_entry == 'per-hosp-stay':
        time_span_name = 'per hospital stay'
    else:
        time_span = parse_time(time_entry)
        time_span_name = format_timedelta(time_span)

    # Remove prefix for medications.
    display_var_name = var_name
    if var_id >= 20000:
        display_var_name = display_var_name.split(' ', 1)[1]

    # Determine the feature name from the function name and time_span.
    feature_characteristics = []
    feature_characteristics.insert(0, display_function_name)
    feature_characteristics.insert(1, time_span_name)
    feature_name = prefix + display_var_name + \
        ((' (' + ' '.join(feature_characteristics)) + ')' if len(feature_characteristics) > 0 else '') + \
        (' [' + var_unit + ']' if var_unit else '')

    return function_name, feature_name


def feat_func_extrapolate(values_in_interval, time_span, stay_start, stay_end):
    # Extrapolate the given values for time_span for the given interval by stay_start and stay_end.
    # Also, normalize the extrapolated values per day.
    sum_interval = sum(values_in_interval) if values_in_interval.shape[0] > 0 else np.nan
    if sum_interval == np.nan:
        return sum_interval
    elif stay_end - stay_start >= time_span:
        # Full time interval covered by values.
        return sum_interval * (datetime.timedelta(days=1) / time_span)
    else:
        # Interval since stay start too short, so extrapolate.
        return (sum_interval * (time_span / (stay_end - stay_start))) * (datetime.timedelta(days=1) / time_span)


if __name__ == '__main__':
    main()
