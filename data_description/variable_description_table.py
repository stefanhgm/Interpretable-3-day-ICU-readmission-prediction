import argparse
import json
import os

import pandas as pd
from joblib import Parallel, delayed
from pandas.core.dtypes.common import is_string_dtype
from psycopg2 import sql

from data_description.stays_description_table import dict_to_string
from helper.io import read_item_processing_descriptions_from_excel
from helper.processing import get_pid_case_ids_map, format_timedelta
from helper.util import output_wrapper, format_timedelta_opt_days
from preprocessing.step5_generate_features_for_stays import decode_feature_time_span
from research_database.research_database_communication import ResearchDBConnection


def main():
    parser = argparse.ArgumentParser(description='Create table for all variables.')
    parser.add_argument('password', type=str, help='Database password.')
    parser.add_argument('item_overview', type=str, help='Description of all PDMS items and generated variables.')
    parser.add_argument('--output_path', type=str, help='Directory to store generated table.')
    args, _ = parser.parse_known_args()

    db_conn = ResearchDBConnection(args.password)
    stays = db_conn.read_table_from_db('stays')
    stays_pid_case_ids = get_pid_case_ids_map(db_conn, stays)
    incl_cases = set(stays['fallnummer'].tolist())

    item_processing = read_item_processing_descriptions_from_excel(args.item_overview)

    # Parse "included" items and merged items as specified by variable description.
    # Possible to add custom filter here. (item_processing['feature_generation'].str.contains('saps_'))
    variables = item_processing.loc[
        ((item_processing['decision'] == 'included') | (item_processing['id'] >= 10000))]
    print(f"Found {variables.shape[0]} variables.")

    variables = variables.loc[:, ['id', 'variable_name', 'type', 'unit', 'feature_generation']]
    variables.rename({'feature_generation': 'features'}, axis=1, inplace=True)

    # Add feature methods.
    variables['features'] = variables['features'].apply(lambda val: json.loads(val))

    variables.reset_index(inplace=True, drop=True)
    # Add statistics in parallel fashion.
    results = Parallel(n_jobs=40)(delayed(parallel_variable_stats)(args.password, idx, variables, incl_cases,
                                                                   stays_pid_case_ids) for idx in list(variables.index))
    # Serial processing for debug purpose.
    # results = [parallel_variable_stats(args.password, idx, variables, incl_cases, patient_cases_map)
    #            for idx in list(variables.index)]

    for idx, _ in variables.iterrows():
        variables.loc[idx, 'non-empty during feat interval'] = str(int(results[idx][5]))
        variables.loc[idx, 'median sampling interval'] = results[idx][4]
        variables.loc[idx, 'statistics during feat interval'] = results[idx][3]
        variables.loc[idx, 'w/out per-patient'] = results[idx][0]
        variables.loc[idx, 'w/out per-hosp-stay'] = results[idx][1]
        variables.loc[idx, 'w/out per-icu_stay'] = results[idx][2]
        variables.loc[idx, 'Imputation value'] = '-'
        variables.loc[idx, 'generated features'] = str(results[idx][6])

        if variables.loc[idx, 'id'] >= 20000:
            variables.loc[idx, 'variable_name'] = variables.loc[idx, 'variable_name'].split(' ', 1)[1]

    # Sort static variables to the top.
    variables.loc[variables['features'].str.join(',').str.contains('static,|static$'), 'ordering'] = 0
    variables.loc[variables['features'].str.
                  join(',').str.contains('static_per-hosp-stay,|static_per-hosp-stay$'), 'ordering'] = 1
    variables.loc[variables['features'].str.
                  join(',').str.contains('static_per-icu-stay,|static_per-icu-stay$'), 'ordering'] = 2
    variables.loc[variables['features'].str.
                  join(',').str.contains('intervention_per-icu-stay,|intervention_per-icu-stay$'), 'ordering'] = 3
    variables.loc[variables['features'].str.
                  join(',').str.contains('timeseries_low,|timeseries_low$'), 'ordering'] = 4
    variables.loc[variables['features'].str.
                  join(',').str.contains('timeseries_medium,|timeseries_medium$'), 'ordering'] = 5
    variables.loc[variables['features'].str.
                  join(',').str.contains('timeseries_high,|timeseries_high$'), 'ordering'] = 6
    variables.loc[variables['features'].str.join(',').str.contains('flow,|flow$'), 'ordering'] = 7
    variables.loc[variables['features'].str.join(',').str.contains('medication,|medication$'), 'ordering'] = 8

    print(f"Detected static variables: "
          f"{variables.loc[variables['ordering'] == 0, 'variable_name'].to_list()}")
    print(f"Detected static per hospital stay variables: "
          f"{variables.loc[variables['ordering'] == 1, 'variable_name'].to_list()}")
    variables = variables.sort_values(['ordering', 'variable_name'], ascending=[True, True], ignore_index=True)\
        .drop('ordering', axis=1)

    variables.to_csv(os.path.join(args.output_path, 'variable_description.csv'),
                     sep='\t', encoding='utf-8', header=True, index=False)


@output_wrapper
def parallel_variable_stats(password, idx, variables, incl_cases, stays_pid_case_ids):
    # Very similar to feature processing determine number of stays w/out value per_patient, per-hosp-stay, per-icu-stay.
    db_conn = ResearchDBConnection(password)
    stays = db_conn.read_table_from_db('stays')
    var_id, var_name, var_type = variables.loc[idx, ['id', 'variable_name', 'type']]
    var_feats = variables.loc[idx, 'features']

    # Read values of included patients for a variable.
    all_cases_of_all_patients = [item for sublist in list(stays_pid_case_ids.values()) for item in sublist]
    values = db_conn.read_values_from_db(var_name, var_type, case_id=all_cases_of_all_patients)
    # Derive datatype boolean a (treated as numerical) if only "true", "false" as strings.
    non_null = ~(values['data'].isnull())
    try:
        if is_string_dtype(values.loc[non_null, 'data']) and\
                all(values.loc[non_null, 'data'].str.lower().isin(['true', 'false'])):
            values.loc[non_null, 'data'] =\
                values.loc[non_null, 'data'].str.lower().replace({'true': True, 'false': False})
    except AttributeError:
        # In case no all values are strings.
        pass

    no_values = {'per-patient': [], 'per-hosp-stay': [], 'per-icu-stay': []}
    for idx, _ in stays.iterrows():
        # Code copied from feature generation.
        icu_start = stays.loc[idx, 'intvon']
        icu_end = stays.loc[idx, 'intbis']
        td_icu_stay_start = icu_end - icu_start
        td_hosp_stay_start = icu_end - stays.loc[idx, 'ukmvon']
        patient_cases = stays_pid_case_ids[stays.loc[idx, 'patientid']]
        patient_values_mask = (values['patientid'].isin(patient_cases))

        for time_entry in ['per-patient', 'per-hosp-stay', 'per-icu-stay']:
            time_span = decode_feature_time_span(time_entry, td_icu_stay_start, td_hosp_stay_start)
            # Determine all patient values for a given time span.
            patient_values_in_time_span =\
                values.loc[patient_values_mask & (values['displaytime'] >= icu_end - time_span) &
                           (values['displaytime'] <= icu_end)].copy()

            # Remove nan values, no feature function needs them.
            patient_values_in_time_span.dropna(inplace=True)

            if patient_values_in_time_span['data'].shape[0] == 0:
                no_values[time_entry].append(stays.loc[idx, 'id'])

    for time_span in ['per-patient', 'per-hosp-stay', 'per-icu-stay']:
        if len(no_values[time_span]) < 250:
            print('\t\t', time_span, no_values[time_span])

    num_wout_per_patient = len(no_values['per-patient'])
    num_wout_per_hosp_stay = len(no_values['per-hosp-stay'])
    num_wout_per_icu_stay = len(no_values['per-icu-stay'])
    print(f"Process variable: {var_name} ({num_wout_per_patient}, {num_wout_per_hosp_stay}, {num_wout_per_icu_stay}).")

    # Determine statistics for all relevant values.
    # Keep all values of a patient (static per hosp stay) or stay (static per stay) or only the ones of an icu stay.
    if 'static' in var_feats:
        values.drop(values[~(values['patientid'].isin(all_cases_of_all_patients))].index, inplace=True)
    elif 'static_per-hosp-stay' in var_feats:
        values.drop(values[~(values['patientid'].isin(incl_cases))].index, inplace=True)
    else:
        # Mark all values of included cases and in relevant time as true.
        values['included'] = False
        for idx, _ in stays.iterrows():
            values.loc[(values['patientid'] == stays.loc[idx, 'fallnummer']) &
                       (values['displaytime'] >= stays.loc[idx, 'intvon']) &
                       (values['displaytime'] <= stays.loc[idx, 'intbis']), 'included'] = True
        values.drop(values[~(values['included'])].index, inplace=True)
        values.drop('included', axis=1, inplace=True)
    non_empty_during_interval = values['data'].notna().sum()
    stats = []
    if values.shape[0] == 0:
        stats = ["no values"]
    elif values['data'].isnull().all():
        stats = ["all unknown"]
    elif var_type == 'continuous':
        stats.append(f"median: {values['data'].median():.2f}")
        stats.append(f"min: {values['data'].min():.2f}")
        stats.append(f"max: {values['data'].max():.2f}")
        # stats.append(f"unk: {values['data'].isna().sum()}")
    elif var_type == 'categorical':
        stats.append(dict_to_string(values['data'].value_counts(dropna=False).to_dict(), sep=','))
    else:
        raise Exception("Unknown variable type.")

    # Determine median sampling interval for values.
    query = sql.SQL("select variablename, percentile_cont(0.5) within group (order by displaytime_interval) "
                    "from (select patientid, variablename, displaytime - lag(displaytime) over "
                    "(partition by patientid, variablename order by displaytime) as displaytime_interval "
                    "from values v where variablename = %s) as recording_intervals group by variablename;")
    result = pd.read_sql_query(query, db_conn.get_cur_conn()[1], params=(str(var_name),))
    if result.shape[0] == 0 or result.loc[0, 'percentile_cont'] is None:
        median_sampling_interval = '-'
    else:
        median_interval = result.loc[0, 'percentile_cont']
        median_sampling_interval = format_timedelta_opt_days(median_interval)

    # Determine all associated features with number of missing values.
    query = sql.SQL("select featurename, "
                    "least(sum(case when numericvalue is null then 1 else 0 end), "
                    "sum(case when textvalue is null then 1 else 0 end)) empty, count(*) total "
                    "from inputs i where featurename like %s group by featurename;")
    if var_id >= 20000:
        var_name = var_name.split(' ', 1)[1]
    result = pd.read_sql_query(query, db_conn.get_cur_conn()[1], params=(str(var_name + '%'),))
    result['featurename'] = \
        result['featurename'].map(lambda x: ((x[len(var_name) + 1:]).replace('(', '')).replace(')', ''))
    result.sort_values('featurename', inplace=True)
    features_string = ""
    for idx, feature in result.iterrows():
        assert feature['total'] == 15589
        features_string += feature['featurename'] + " " + str(feature['empty']) + "/" + str(feature['total']) + "\n"

    return num_wout_per_patient, num_wout_per_hosp_stay, num_wout_per_icu_stay, ','.join(stats),\
        median_sampling_interval, non_empty_during_interval, features_string


if __name__ == '__main__':
    main()
