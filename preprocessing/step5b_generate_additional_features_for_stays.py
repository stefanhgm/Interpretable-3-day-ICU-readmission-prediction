"""
Script to create static features for the stays from different sources than the variables.
"""
import argparse

from joblib import Parallel, delayed

from helper.util import output_wrapper
from research_database.research_database_communication import ResearchDBConnection

import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Extract static features.')
    parser.add_argument('password', type=str, help='Database password.')
    parser.add_argument('--variable_names', nargs='+', help='Only convert specified variables.', default='')
    args, _ = parser.parse_known_args()

    static_feat_processing = [
        create_static_is_3d_icu_readmission,
        create_static_icu_station,
        create_static_icu_length_of_stay_days,
        create_static_length_of_stay_before_icu_days
    ]

    print(f"Detected {len(static_feat_processing)} static features to generate.")
    Parallel(n_jobs=4)(delayed(fn)(args.password) for fn in static_feat_processing)
    # To process in serial manner:
    # [parallel_feature_processing(args.password, idx, feat_processing) for idx in list(feat_processing.index.values)]


@output_wrapper
def create_static_is_3d_icu_readmission(password):
    db_conn = ResearchDBConnection(password)
    feat_name, stays = static_is_3d_icu_readmission(password)
    db_conn.write_features_and_inputs_to_db(feat_name, stays)


@output_wrapper
def static_is_3d_icu_readmission(password):
    feat_name = 'Is 3-day ICU readmission'
    db_conn = ResearchDBConnection(password)
    stays = db_conn.read_table_from_db('stays')
    assert stays.loc[stays['label']].shape[0] > 0
    stays['featurename'] = feat_name
    stays.sort_values(['patientid', 'intvon'], inplace=True, ignore_index=True)
    stays[['prev_label', 'prev_patientid']] = stays[['label', 'patientid']].shift(periods=1)
    stays.loc[stays['patientid'] != stays['prev_patientid'], 'prev_label'] = False
    stays['numericvalue'] = 0
    stays.loc[stays['prev_label'], 'numericvalue'] = 1
    stays.rename(columns={'id': 'stayid'}, inplace=True)
    stays.drop(stays.columns.difference(['stayid', 'featurename', 'numericvalue']), axis=1, inplace=True)
    print(f"Created {stays.shape[0]} features for {feat_name}.")
    return feat_name, stays


@output_wrapper
def create_static_icu_station(password):
    db_conn = ResearchDBConnection(password)
    feat_name, stays = static_icu_station(password)
    db_conn.write_features_and_inputs_to_db(feat_name, stays)


@output_wrapper
def static_icu_station(password):
    feat_name = 'ICU station'
    db_conn = ResearchDBConnection(password)
    stays = db_conn.read_table_from_db('stays')
    stays['featurename'] = feat_name
    stays['textvalue'] = (stays['station'].str.split(pat=';')).str[-1]
    stays.rename(columns={'id': 'stayid'}, inplace=True)
    stays.drop(stays.columns.difference(['stayid', 'featurename', 'textvalue']), axis=1, inplace=True)
    print(f"Created {stays.shape[0]} features for {feat_name}.")
    return feat_name, stays


@output_wrapper
def create_static_icu_length_of_stay_days(password):
    db_conn = ResearchDBConnection(password)
    feat_name, stays = static_icu_length_of_stay_days(password)
    db_conn.write_features_and_inputs_to_db(feat_name, stays)


@output_wrapper
def static_icu_length_of_stay_days(password):
    feat_name = 'ICU length of stay [days]'
    db_conn = ResearchDBConnection(password)
    stays = db_conn.read_table_from_db('stays')
    stays['featurename'] = feat_name
    # LOS in days based on integer value of LOS in hours.
    stays['numericvalue'] = ((stays['intbis'] - stays['intvon']) / np.timedelta64(1, 'h')).astype('int') / 24.
    stays.rename(columns={'id': 'stayid'}, inplace=True)
    stays.drop(stays.columns.difference(['stayid', 'featurename', 'numericvalue']), axis=1, inplace=True)
    print(f"Created {stays.shape[0]} features for {feat_name}.")
    return feat_name, stays


@output_wrapper
def create_static_length_of_stay_before_icu_days(password):
    db_conn = ResearchDBConnection(password)
    feat_name, stays = static_length_of_stay_before_icu_days(password)
    db_conn.write_features_and_inputs_to_db(feat_name, stays)


@output_wrapper
def static_length_of_stay_before_icu_days(password):
    feat_name = 'Length of stay before ICU [days]'
    db_conn = ResearchDBConnection(password)
    stays = db_conn.read_table_from_db('stays')
    stays['featurename'] = feat_name
    # LOS in days based on integer value of LOS in hours.
    stays['numericvalue'] = ((stays['intvon'] - stays['ukmvon']) / np.timedelta64(1, 'h')).astype('int') / 24.
    stays.rename(columns={'id': 'stayid'}, inplace=True)
    stays.drop(stays.columns.difference(['stayid', 'featurename', 'numericvalue']), axis=1, inplace=True)
    print(f"Created {stays.shape[0]} features for {feat_name}.")
    return feat_name, stays


if __name__ == '__main__':
    main()
