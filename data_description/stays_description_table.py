import argparse
import datetime as dt
import os

import numpy as np
import pandas as pd

from preprocessing.step6_output_features_and_labels import pivot_inputs_for_features
from research_database.research_database_communication import ResearchDBConnection


def main():
    parser = argparse.ArgumentParser(description='Create table to summarize cohort characteristics.')
    parser.add_argument('password', type=str, help='Database password.')
    parser.add_argument('--output_path', type=str, help='Directory to store generated plots.')
    args, _ = parser.parse_known_args()

    db_conn = ResearchDBConnection(args.password)
    stays = db_conn.read_table_from_db('stays')
    features = db_conn.read_table_from_db('features')
    inputs = db_conn.read_table_from_db('inputs')

    stays = stays.set_index('id')
    feature_output = pivot_inputs_for_features(features, inputs)
    old_num_stays = stays.shape[0]
    stays = stays.join(feature_output, how='inner')
    assert old_num_stays == stays.shape[0]

    # Add additional fields to stays.
    stays['icu_duration'] = stays['intbis'] - stays['intvon']

    stays_false = stays.loc[~stays['label']]
    stays_true = stays.loc[stays['label']]

    # Create table for cohort characteristics.
    table = pd.DataFrame({'Characteristic': [], 'All stays': [], 'Labeled false': [], 'Labeled true': []})
    table = table.append({'Characteristic': 'Number of patients',
                          'All stays': f"{stays.shape[0]} ({stays.shape[0]/stays.shape[0]:.1%})",
                          'Labeled false': f"{stays_false.shape[0]} ({stays_false.shape[0]/stays.shape[0]:.1%})",
                          'Labeled true': f"{stays_true.shape[0]} ({stays_true.shape[0]/stays.shape[0]:.1%})"},
                         ignore_index=True)
    # Age
    age = 'Age (static all data) [years]'
    table = add_mean_sd_to_table(table, stays, stays_false, stays_true, age, 'Mean age (SD)')

    # Gender
    gender = 'Gender (static all data)'
    table = table.append({'Characteristic': 'Gender',
                          'All stays': f"{dict_to_string(stays[gender].value_counts().to_dict())}",
                          'Labeled false': f"{dict_to_string(stays_false[gender].value_counts().to_dict())}",
                          'Labeled true': f"{dict_to_string(stays_true[gender].value_counts().to_dict())}"},
                         ignore_index=True)

    # Station
    station_counts_all = {}
    station_counts_true = {}
    station_counts_false = {}
    for idx, _ in stays.iterrows():
        stations_string = stays.loc[idx, 'station']
        for station in stations_string.split(';'):
            station_counts_all[station] = station_counts_all.get(station, 0) + 1
            if stays.loc[idx, 'label']:
                station_counts_true[station] = station_counts_true.get(station, 0) + 1
            else:
                station_counts_false[station] = station_counts_false.get(station, 0) + 1
    table = table.append({'Characteristic': 'Stations',
                          'All stays': f"{dict_to_string(station_counts_all)}",
                          'Labeled false': f"{dict_to_string(station_counts_false)}",
                          'Labeled true': f"{dict_to_string(station_counts_true)}"},
                         ignore_index=True)

    # Length of stay
    length_of_stay = 'icu_duration'
    table =\
        add_mean_sd_to_table_td(table, stays, stays_false, stays_true, length_of_stay, 'Mean length of ICU stay (SD)')

    table.to_csv(os.path.join(args.output_path, 'cohort_description.csv'), sep=',', header=True, index=False)


def add_mean_sd_to_table(table, stays, stays_false, stays_true, field, name):
    table = table.append({'Characteristic': name,
                          'All stays': f"{stays[field].mean():.2f} ({stays[field].std():.2f})",
                          'Labeled false': f"{stays_false[field].mean():.2f} ({stays_false[field].std():.2f})",
                          'Labeled true': f"{stays_true[field].mean():.2f} ({stays_true[field].std():.2f})"},
                         ignore_index=True)
    return table


def add_mean_sd_to_table_td(table, stays, stays_false, stays_true, field, name):
    # From: https://stackoverflow.com/questions/44616546/
    # finding-the-mean-and-standard-deviation-of-a-timedelta-object-in-pandas-df
    ns = stays[field].astype(np.int64)
    ns_false = stays_false[field].astype(np.int64)
    ns_true = stays_true[field].astype(np.int64)
    table = table.append({'Characteristic': name,
                          'All stays': f"{chop_ms(pd.to_timedelta(ns.mean()))} "
                                       f"({chop_ms(pd.to_timedelta(ns.std()))}))",
                          'Labeled false': f"{chop_ms(pd.to_timedelta(ns_false.mean()))} "
                                           f"({chop_ms(pd.to_timedelta(ns_false.std()))})",
                          'Labeled true': f"{chop_ms(pd.to_timedelta(ns_true.mean()))} "
                                          f"({chop_ms(pd.to_timedelta(ns_true.std()))})"},
                         ignore_index=True)
    return table


def chop_ms(delta):
    # From: https://stackoverflow.com/questions/18470627/how-do-i-remove-the-microseconds-from-a-timedelta-object
    return delta - dt.timedelta(microseconds=delta.microseconds)


def dict_to_string(dictionary, sep=' '):
    return sep.join([f'{key} (n={value})' for key, value in dictionary.items()])


if __name__ == '__main__':
    main()
