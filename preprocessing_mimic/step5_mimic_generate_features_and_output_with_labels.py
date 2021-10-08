"""
Generate features for most relevant EBM risk functions directly from MIMIC-IV database.
"""
import argparse
import datetime

import json
import numpy as np
import os
import pandas as pd
from joblib import Parallel, delayed

from helper.io import read_item_processing_descriptions_from_excel
from helper.util import output_wrapper
from preprocessing.step5_generate_features_for_stays import decode_feature_time_span, feat_func_map, \
    feat_func_interval_map, feat_func_interval_start_end_map, feat_func_with_times_map, feat_func_with_times_span_map, \
    decode_func_and_feat_name, feat_methods
from research_database.research_database_communication import ResearchDBConnection, resolve_data_column_name, \
    MIMICDBConnection

# Min/max values derived from concept creation routines and when necessary as seen during RF review.
mimic_features_ebm = [
    # Variable name,                interval, function,     min,    max,    imputation, MIMIC mimic_icu.d_items item id
    # own min/max
    # removed after inspections
    # ('CK',                          '7d',   'median',       1.,     10000., 201.,   [225634]),
    ('CK',                          '7d',   'min',          1.,     10000., 201.,   [225634]),
    ('CK-MB',                       '3d',   'median',       1.,     10000., None,   [227445]),
    ('CK-MB',                       '3d',   'max',          1.,     10000., None,   [227445]),
    # own min/max
    ('Chloride',                    '3d',   'trend',        1.,     200.,   103.,   [220602]),
    ('Chloride',                    '1d',   'min',          1.,     200.,   103.,   [220602]),
    # own min/max
    # removed after inspections
    # ('PTT',                         '3d',   'min',          1.,     500.,   34.3,   [227466]),
    # ('PTT',                         '3d',   'max',          1.,     500.,   34.3,   [227466]),
    # ('PTT',                         '1d',   'max',          1.,     500.,   34.3,   [227466]),
    # ('PTT',                         '7d',   'min',          1.,     500.,   34.3,   [227466]),
    # ('PTT',                         '7d',   'max',          1.,     500.,   34.3,   [227466]),
    # Used both arterial and venous BGA together as in original cohort. Use own min/max against artifacts.
    ('pH',                          '1d',   'median',       7.0,    7.8,    7.39,   [220274, 223830]),
    ('pH',                          '3d',   'median',       7.0,    7.8,    7.39,   [220274, 223830]),
    ('pH',                          '3d',   'trend',        7.0,    7.8,    7.39,   [220274, 223830]),
    ('pH',                          '1d',   'iqr',          7.0,    7.8,    7.39,   [220274, 223830]),
    ('Blood volume out',            '7d',   'extrapolate',  None,   None,   None,   [226626, 226629]),
    ('Blood volume out',            '3d',   'extrapolate',  None,   None,   None,   [226626, 226629]),
    # own min/max
    ('Hematocrit',                  '3d',   'max',          1.,     100.,   28.7,   [220545]),
    ('Hematocrit',                  '12h',  'median',       1.,     100.,   28.7,   [220545]),
    # Only "Arterial Base Excess"
    ('BE',                          '12h',  'median',       -25.,   25.,    0.0,    [220545]),
    ('BE',                          '3d',   'trend',        -25.,   25.,    0.0,    [220545]),
    # removed after inspections
    # ('BE',                          '3d',   'iqr',          -25.,   25.,    0.0,    [220545]),
    ('BE',                          '12h',  'min',          -25.,   25.,    0.0,    [220545]),
    ('BE',                          '1d',   'iqr',          -25.,   25.,    0.0,    [220545]),
    # own min/max
    ('Phosphate',                   '1d',   'max',          0.01,   500.,   None,   [225677]),
    ('Phosphate',                   '7d',   'min',          0.01,   500.,   None,   [225677]),
    ('Potassium',                   '1d',   'median',       None,   None,   None,   [227442]),
    # Use endtime of invasive and non-invasive ventilation.
    ('Is on automatic ventilation', 'per-icu-stay', 'true-until-discharge', None, None, None, [225792, 225794]),
    ('RAS scale',                   '3d',   'max',          None,   None,   0.0,    [228096]),
    # removed after inspections
    # ('RAS scale',                   '12h',  'trend',        None,   None,   0.0,    [228096]),
    ('RAS scale',                   '1d',   'max',          None,   None,   0.0,    [228096]),
    # Arterial, Venous, Mixed Venous, own min/max
    ('pO2',                         '12h',  'min',          1.,     200.,   102.,   [220224, 226063, 227516]),
    ('pO2',                         '12h',  'iqr',          1.,     200.,   102.,   [220224, 226063, 227516]),
    # Arterial, mixed venous, central venous, pulseoxymetry,
    ('O2 saturation',               '12h',  'min',          0.1,    100.,   97.,    [220227, 225674, 227686, 220277]),
    # Ionized as for original cohort. own min/max
    # removed after inspections
    # ('Calcium',                     '3d',   'trend',        0.1,    30.,    1.12,   [225667]),
    ('Calcium',                     '1d',   'max',          0.1,    30.,    1.12,   [225667]),
    # Min/max from derived vital signs
    ('Heart rate',                  '4h',   'min',          1,      299,    None,   [220045]),
    ('Heart rate',                  '1d',   'min',          1,      299,    None,   [220045]),
    ('Heart rate',                  '4h',   'iqr',          1,      299,    None,   [220045]),
    # In contrast to original cohort only two items (Temp. celcius, Blood temperature) and no complex merging procedure.
    # removed after inspections
    # ('Body core temperature',       '1d',   'min',          10.1,   49.9,   37.,    [223762, 226329]),
    # ('Body core temperature',       '4h',   'min',          10.1,   49.9,   37.,    [223762, 226329]),
    ('Body core temperature',       '1d',   'median',       10.1,   49.9,   37.,    [223762, 226329]),
    ('Body core temperature',       '1d',   'trend',        10.1,   49.9,   37.,    [223762, 226329]),
    # Arterial, Venous, own min/max
    ('pCO2',                        '1d',   'median',       1.,     100.,   41.,    [220235, 226062]),
    ('pCO2',                        '1d',   'iqr',          1.,     100.,   41.,    [220235, 226062]),
    ('pCO2',                        '3d',   'min',          1.,     100.,   41.,    [220235, 226062]),
    # own min/max
    # removed after inspections
    # ('Leucocytes',                  '3d',   'trend',        0.1,    100.,   10.8,   [220546]),
    ('Leucocytes',                  '1d',   'median',       0.1,    100.,   10.8,   [220546]),
    ('Leucocytes',                  '3d',   'iqr',          0.1,    100.,   10.8,   [220546]),
    ('Blood Urea Nitrogen',         '3d',   'min',          None,   None,   None,   [225624]),
    # Urine and GU Irrigant Out, GU Irrigant/Urine Volume Out, OR, PACU
    ('Urine volume out',            '1d',   'extrapolate',  None,   None,   None,   [226566, 227489, 226627, 226631]),
    ('Urine volume out',            '7d',   'extrapolate',  None,   None,   None,   [226566, 227489, 226627, 226631]),
    ('Bilirubin total',             '7d',   'max',          None,   None,   None,   [225690]),
    # own min/max
    ('Lactate',                     '3d',   'max',          0.01,   50.,   1.8,    [225668]),
    ('Lactate',                     '12h',  'min',          0.01,   50.,   1.8,    [225668]),
    # own min/max
    ('Sodium',                      '3d',   'trend',        1.,     300.,   139.,   [220645]),
    ('Sodium',                      '3d',   'median',       1.,     300.,   139.,   [220645]),
    # own min/max
    ('Hemoglobin',                  '3d',   'max',          1.,     30.,   9.5,    [220228]),
    # Used all BP values of category "Routine Vital Signs"
    ('Diastolic blood pressure',    '1d',   'median',       None,   None,   61,     [227242, 224643, 225310, 220180,
                                                                                     220051]),
    ('Mean blood pressure',         '4h',   'median',       None,   None,   77,     [225312, 220181, 220052]),
    ('Mean blood pressure',         '12h',  'median',       None,   None,   77,     [225312, 220181, 220052]),
    ('Systolic blood pressure',     '12h',  'iqr',          None,   None,   117,    [227243, 224167, 225309, 220179,
                                                                                     220050]),
    # Used "Respiratory Rate", "RR spontaneous" and "RR total"
    ('Estimated respiratory rate',  '1d',   'median',       None,   None,   None,   [220210, 224689, 224690]),
    # own min/max
    ('Thrombocytes',                '7d',   'trend',        1.,     1000.,  176.,   [227457]),
    ('Glucose',                     '3d',   'median',       10.,    1200.,  127.,   [220621]),
    # Used "Invasive ventilation"
    ('Tubus exists',                'per-icu-stay', 'true-until-discharge', None, None, None, [225792]),
    # own min/max
    ('C-reactive protein',          '3d',   'max',          0.01,   60.,    None,   [225792]),
    # No drugs for constipation in icu medications and also only few identified in emar(_detail)
    # Hence, use "Elimination NCP - Interventions" (229134) instead as an indicator for constipation.
    ('Drugs for constipation',      '1d',   'last',         None,   None,   None,   [229134]),

    # Special routines necessary:
    ('Age',                         'per-patient', 'last',  None,   None,   None,   []),
    ('Length of stay before ICU',   'per-patient', 'last',  None,   None,   None,   []),
    ('paO2/FiO2',                   '1d',     'median',     None,   None,   210.,   []),
    ('paO2/FiO2',                   '3d',     'trend',      None,   None,   210.,   []),
    ('paO2/FiO2',                   '3d',     'median',     None,   None,   210.,   []),
    # In MIMIC cohort always 1.
    ('MetHb',                       '12h',    'min',        None,   None,   1.,     []),
    ('GCS score',                   '3d',     'min',        None,   None,   15,     []),
    # own min/max
    ('eGFR',                        '7d',     'trend',      0.,     300.,   68.4,   []),
    ('Gamma-GT',                    '7d',     'median',     None,   None,   None,   []),
    ('Antithrombotic agents prophylactic dosage', 'per-icu-stay', 'true-until-discharge', None, None, None, []),

    # Not in MIMIC DB
    # 'Procalcitonin (max 7d) [ng/mL]'
    # 'RHb (median 12h)'
]

mimic_variables_gbm = [
    'Age', 'Antithrombotic agents prophylactic dosage', 'BE', 'Bilirubin total', 'Blood Urea Nitrogen',
    'Blood volume out', 'Body core temperature', 'C-reactive protein', 'CK', 'CK-MB', 'Calcium', 'Chloride',
    'Diastolic blood pressure', 'Drugs for constipation', 'Estimated respiratory rate', 'GCS score', 'Gamma-GT',
    'Glucose', 'Heart rate', 'Hematocrit', 'Hemoglobin', 'Is on automatic ventilation', 'Lactate',
    'Length of stay before ICU', 'Leucocytes', 'Mean blood pressure', 'MetHb', 'O2 saturation', 'Phosphate',
    'Potassium', 'RAS scale', 'Sodium', 'Systolic blood pressure', 'Thrombocytes', 'Tubus exists', 'Urine volume out',
    'eGFR', 'pCO2', 'pH', 'pO2', 'paO2/FiO2'
]


def main():
    parser = argparse.ArgumentParser(description='Extract features.')
    parser.add_argument('password', type=str, help='Database password.')
    parser.add_argument('item_overview', type=str, help='Description of all PDMS items and generated variables.')
    parser.add_argument('output_file', type=str, help='File to store output data.')
    parser.add_argument('--gbm', action='store_true', help='Create all features for included variables.')
    parser.add_argument('--force', action='store_true', help='Force to override output file.')
    args, _ = parser.parse_known_args()

    db_conn = ResearchDBConnection(args.password)
    mimic_stays = db_conn.read_table_from_db('mimic_stays')
    # Add anchor year to stays to use it for the output.
    mimic_db_conn = MIMICDBConnection(args.password)
    mimic_patients = mimic_db_conn.read_table_from_db('mimic_core', 'patients')[['subject_id', 'anchor_year_group']]
    old_num_mimic_stays = mimic_stays.shape[0]
    mimic_stays = pd.merge(mimic_stays, mimic_patients, how='left', on='subject_id')
    mimic_stays['anchor_year_group'] = mimic_stays['anchor_year_group'].map(
        {'2008 - 2010': pd.to_datetime('2008', format='%Y'),
         '2011 - 2013': pd.to_datetime('2011', format='%Y'),
         '2014 - 2016': pd.to_datetime('2014', format='%Y'),
         '2017 - 2019': pd.to_datetime('2017', format='%Y')})
    assert mimic_stays.shape[0] == old_num_mimic_stays
    assert (mimic_stays.loc[mimic_stays['anchor_year_group'].isna()]).shape[0] == 0

    mimic_features = mimic_features_ebm
    # GBM: If features for GBM, generate all possible features for each variable.
    if args.gbm:
        mimic_features_gbm = []
        item_processing = read_item_processing_descriptions_from_excel(args.item_overview)
        item_processing =\
            item_processing.loc[(item_processing['decision'] == 'included') | (item_processing['id'] >= 10000)]
        for var in mimic_variables_gbm:
            if var == 'Length of stay before ICU':
                mimic_features_gbm.append(('Length of stay before ICU', 'per-patient', 'last', None, None, None, []))
                continue
            # Use feature vector for EBM as blueprint.
            feat_tuple = None if var != 'PTT' else ('PTT', '', '', 1., 500., 34.3, [227466])
            for mimic_feat_tuple in mimic_features_ebm:
                if mimic_feat_tuple[0] == var:
                    feat_tuple = mimic_feat_tuple
                    break
            assert feat_tuple is not None
            # Generate all necessary feature functions for this variable.
            if var == 'Drugs for constipation':
                feat_generation = 'medication'
            else:
                feat_generation =\
                    item_processing.loc[item_processing['variable_name'] == var, 'feature_generation'].iloc[0]
                feat_generation = json.loads(feat_generation)[0]
            for interval in feat_methods[feat_generation].keys():
                for fn in feat_methods[feat_generation][interval]:
                    new_feat_tuple =\
                        (feat_tuple[0], interval, fn, feat_tuple[3], feat_tuple[4], feat_tuple[5], feat_tuple[6])
                    mimic_features_gbm.append(new_feat_tuple)
        mimic_features = mimic_features_gbm
        print(f"For GBM data determined {len(mimic_features_gbm)} features for {len(mimic_variables_gbm)} variables.")

    # Process all specified features described in the variable description.
    processed_feats = Parallel(n_jobs=40)(delayed(mimic_parallel_feature_processing)(args.password, mimic_stays, feat)
                                          for feat in mimic_features)
    # To process in serial manner:
    # [parallel_feature_processing(args.password, var, stays_pid_case_ids) for _, var in variables.iterrows()]

    # Write generated features into columns.
    output = pd.DataFrame([])
    for processed_feat in processed_feats:
        for feat_name, feat_values in processed_feat.items():
            output[feat_name] = feat_values

    # Order columns alphabetically to get reproducible outputs.
    output = output.reindex(sorted(output.columns), axis=1)

    output.loc[:, 'year_icu_out'] = mimic_stays['anchor_year_group']
    output.loc[:, 'label'] = mimic_stays['label']

    assert output.shape[0] == mimic_stays.shape[0]
    assert output.shape[1] - 2 == len(mimic_features)
    if not args.gbm:
        assert output.shape[1] - 2 == 66
    else:
        assert output.shape[1] - 2 == 515

    print('Write features and labels to', args.output_file)
    print("")
    if args.force and os.path.exists(args.output_file):
        os.remove(args.output_file)
    output.to_csv(args.output_file, sep=',', header=True, index=False)


@output_wrapper
def mimic_parallel_feature_processing(password, mimic_stays, feature):
    mimic_db_conn = MIMICDBConnection(password)
    feat_name, feat_interval, feat_func_name, feat_min, feat_max, feat_imputation, feat_item_ids = feature

    # Read values of included patients for a variable.
    # all_cases_of_all_patients = [item for sublist in list(mimic_stays_pid_case_ids.values()) for item in sublist]
    if len(feat_item_ids) > 0:
        mimic_items = mimic_db_conn.read_table_from_db('mimic_icu', 'd_items')
        feat_values = pd.DataFrame([])
        for feat_item_id in feat_item_ids:
            item = (mimic_items.loc[mimic_items['itemid'] == feat_item_id]).iloc[0]
            item_values = mimic_db_conn.read_values_from_db(feat_item_id, item['linksto'],
                                                            case_id=mimic_stays['subject_id'].tolist())
            feat_values = pd.concat([feat_values, item_values], axis=0)
        print(f"\tProcess variable {feat_name} (id: {feat_item_ids}) for {feat_values.shape[0]} values with function "
              f"{feat_func_name}.")
    else:
        feat_values = custom_feat_values(mimic_db_conn, mimic_stays, feature)

    # Some custom processing.
    # For true until discharge data only serves as an indicator.
    if feat_func_name == 'true-until-discharge':
        feat_values['data'] = True
    if feat_name == 'Drugs for constipation':
        feat_values['data'] = True

    if feat_min != None:
        old_num_values = feat_values.shape[0]
        feat_values.drop(feat_values[feat_values['data'] < feat_min].index, inplace=True)
        print(f"\tRemoved {old_num_values - feat_values.shape[0]} smaller than minimum: {feat_min}.")
    if feat_max != None:
        old_num_values = feat_values.shape[0]
        feat_values.drop(feat_values[feat_values['data'] > feat_max].index, inplace=True)
        print(f"\tRemoved {old_num_values - feat_values.shape[0]} larger than maximum: {feat_max}.")

    # For each stay consider all recordings of patient before stay end and apply all feature methods.
    feat_inputs = []  # result
    for idx, _ in mimic_stays.iterrows():
        icu_start = mimic_stays.loc[idx, 'icu_intime']
        icu_end = mimic_stays.loc[idx, 'icu_outtime']
        td_icu_stay_start = icu_end - icu_start
        td_hosp_stay_start = icu_end - mimic_stays.loc[idx, 'admittime']
        patient_values_mask = (feat_values['subject_id'] == mimic_stays.loc[idx, 'subject_id'])

        # Determine all patient values for a given time span.
        time_span = decode_feature_time_span(feat_interval, td_icu_stay_start, td_hosp_stay_start)
        patient_values_in_time_span =\
            feat_values.loc[patient_values_mask & (feat_values['charttime'] >= icu_end - time_span) &
                            (feat_values['charttime'] <= icu_end)].copy()
        # Remove nan values, no feature function needs them.
        patient_values_in_time_span.dropna(inplace=True)
        # Impute selected timeseries variables.
        if patient_values_in_time_span['data'].shape[0] == 0 and feat_imputation is not None:
            patient_values_in_time_span = patient_values_in_time_span.append(
                {'subject_id': mimic_stays.loc[idx, 'subject_id'], 'charttime': icu_end,
                 'data': feat_imputation}, ignore_index=True)

        if patient_values_in_time_span['data'].shape[0] > 0:
            # Set time relative to stay end.
            patient_values_in_time_span.loc[:, 'time_to_stay_end'] = \
                patient_values_in_time_span.loc[:, 'charttime'] - icu_end
            assert all(patient_values_in_time_span['time_to_stay_end'] >= -time_span)
            assert all(patient_values_in_time_span['time_to_stay_end'] <= datetime.timedelta(seconds=0))

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
        feat_inputs.append(feat_value)

    print(f"\t\tProcessed features: {feat_name}.")
    _, feat_gen_name = decode_func_and_feat_name(0, feat_name, None, feat_interval, feat_func_name)
    return {feat_gen_name: feat_inputs}


def custom_feat_values(mimic_db_conn, mimic_stays, feature):
    feat_name, feat_interval, feat_func_name, feat_min, feat_max, feat_imputation, feat_item_ids = feature

    if feat_name == 'Age':
        values = mimic_db_conn.\
            read_table_from_db('mimic_derived', 'icustay_detail')[['subject_id', 'icu_intime', 'admission_age']]
        values = values.rename(columns={'icu_intime': 'charttime', 'admission_age': 'data'})

    elif feat_name == 'Length of stay before ICU':
        values = mimic_db_conn\
            .read_table_from_db('mimic_derived', 'icustay_detail')[['subject_id', 'icu_intime', 'admittime']]
        values['data'] = values['icu_intime'] - values['admittime']
        values['data'] = values['data'].dt.total_seconds() / (3600 * 24)
        values.loc[values['data'] < 0., 'data'] = 0.
        values.drop('admittime', axis=1, inplace=True)
        values = values.rename(columns={'icu_intime': 'charttime'})

    elif feat_name == 'paO2/FiO2':
        values = mimic_db_conn\
            .read_table_from_db('mimic_derived', 'bg')[['subject_id', 'charttime', 'pao2fio2ratio']]
        values = values.rename(columns={'pao2fio2ratio': 'data'})

    elif feat_name == 'MetHb':
        values = mimic_db_conn\
            .read_table_from_db('mimic_derived', 'bg')[['subject_id', 'charttime', 'methemoglobin']]
        values = values.rename(columns={'methemoglobin': 'data'})

    elif feat_name == 'GCS score':
        values = mimic_db_conn\
            .read_table_from_db('mimic_derived', 'gcs')[['subject_id', 'charttime', 'gcs']]
        values = values.rename(columns={'gcs': 'data'})

    elif feat_name == 'Gamma-GT':
        values = mimic_db_conn\
            .read_table_from_db('mimic_derived', 'enzyme')[['subject_id', 'charttime', 'ggt']]
        values = values.rename(columns={'ggt': 'data'})

    # Anticoagulants in inputevents:
    # Argatroban (225147), Bivalirudin (Angiomax) (225148), Heparin Sodium (225152),
    # Enoxaparin (Lovenox) (225906), Fondaparinux (225908), Heparin Sodium (Prophylaxis) (225975),
    # Heparin Sodium (Impella) (229597), Bivalirudin (Angiomax) (Impella) (229781),
    # Heparin Sodium (CRRT-Prefilter) (230044), Coumadin (Warfarin)
    # Used here: Anticoagulants with ordercategoryname  '10-Prophylaxis (IV)' or '11-Prophylaxis (Non IV)'
    # - Heparin Sodium (Prophylaxis) (225975)
    # - Enoxaparin (Lovenox) (225906)
    # - Fondaparinux (225908)
    # - Coumadin (Warfarin) (225913)
    elif feat_name == 'Antithrombotic agents prophylactic dosage':
        heparin = mimic_db_conn.read_values_from_db(225975, 'inputevents', case_id=mimic_stays['subject_id'].tolist())
        heparin['name'] = 'Heparin prophylaxis'
        enoxa = mimic_db_conn.read_values_from_db(225906, 'inputevents', case_id=mimic_stays['subject_id'].tolist())
        enoxa['name'] = 'Enoxaparin prophylaxis'
        fonda = mimic_db_conn.read_values_from_db(225908, 'inputevents', case_id=mimic_stays['subject_id'].tolist())
        fonda['name'] = 'Fondaparinux prophylaxis'
        coumadin = mimic_db_conn.read_values_from_db(225913, 'inputevents', case_id=mimic_stays['subject_id'].tolist())
        coumadin['name'] = 'Coumadin prophylaxis'
        values = pd.concat([heparin, enoxa, fonda, coumadin], axis=0)
        values = values.loc[(values['data'] != '10-Prophylaxis (IV)') |
                            (values['data'] != '11-Prophylaxis (Non IV)'),
                            ['subject_id', 'charttime', 'name']]
        values = values.rename(columns={'name': 'data'})

    elif feat_name == 'eGFR':
        age_gender_ethnicity = mimic_db_conn.\
            read_table_from_db('mimic_derived', 'icustay_detail')\
            [['subject_id', 'icu_intime', 'gender', 'admission_age', 'ethnicity']]
        age_gender_ethnicity = age_gender_ethnicity.rename(columns={'icu_intime': 'charttime'})
        # Calculate eGFR analogous to original cohort according to ckd-epi formula:
        # Use: https://www.kidney.org/content/ckd-epi-creatinine-equation-2009
        creatinine = mimic_db_conn.read_values_from_db(220615, 'chartevents', case_id=mimic_stays['subject_id'].tolist())
        age_gender_ethnicity.sort_values('charttime', inplace=True)
        creatinine.sort_values('charttime', inplace=True)
        # Merge creatinine with latest age, gender, ethnicity entry
        values = pd.merge_asof(creatinine, age_gender_ethnicity, on='charttime', by='subject_id')
        values['creatinine'] = values['data']
        values.drop(values[(values['creatinine'] <= 0) | (values['creatinine'] > 500)].index, inplace=True)
        values.drop(values[(values['gender'].isna()) | (values['admission_age'].isna())].index, inplace=True)
        values.loc[(values['gender'] == 'F'), ['kappa', 'alpha', 'factor_female']] = [0.7, -0.329, 1.018]
        values.loc[(values['gender'] == 'M'), ['kappa', 'alpha', 'factor_female']] = [0.9, -0.411, 1.]
        values['factor_black'] = 1.
        values.loc[(values['ethnicity'] == 'BLACK/AFRICAN AMERICAN'), 'factor_black'] = 1.159
        values['data'] = values.apply(
            lambda row: 141 * (min(row['creatinine'] / row['kappa'], 1) ** row['alpha']) *
                        (max(row['creatinine'] / row['kappa'], 1) ** (-1.209)) *
                        (0.993 ** row['admission_age']) * row['factor_female'] * row['factor_black'], axis=1)
        values.drop(['gender', 'admission_age', 'ethnicity', 'creatinine', 'kappa', 'alpha', 'factor_female',
                     'factor_black'], axis=1, inplace=True)

    else:
        raise ValueError("Bad feature name.")

    values.drop(values[values['data'].isna()].index, inplace=True)
    return values


if __name__ == '__main__':
    main()
