"""
Script to export features and labels from database into csv file.
"""
import argparse
import os

import pandas as pd

from research_database.research_database_communication import ResearchDBConnection


def main():
    parser = argparse.ArgumentParser(description='Merge feature and labels and output as csv.')
    parser.add_argument('password', type=str, help='Database password.')
    parser.add_argument('output_file', type=str, help='File to store output data.')
    parser.add_argument('--force', action='store_true', help='Force to override output file.')
    args, _ = parser.parse_known_args()

    db_conn = ResearchDBConnection(args.password)
    stays = db_conn.read_table_from_db('stays')
    features = db_conn.read_table_from_db('features')
    inputs = db_conn.read_table_from_db('inputs')

    output = pivot_inputs_for_features(features, inputs)
    assert output.shape[0] == stays.shape[0]
    # Add year until stay lasts and label.
    stays.loc[:, 'year_icu_out'] = pd.to_datetime(stays.loc[:, 'intbis'].dt.year, format="%Y")
    stays = stays.set_index('id')
    output = output.merge(stays[['year_icu_out', 'label']], left_index=True, right_index=True, how='inner')
    assert output.shape[0] == stays.shape[0]
    assert output.shape[1] - 2 == features.shape[0]

    print('Write features and labels to', args.output_file)
    if args.force and os.path.exists(args.output_file):
        os.remove(args.output_file)
    output.to_csv(args.output_file, sep=',', header=True, index=False)


def pivot_inputs_for_features(features, inputs):
    features.sort_values(['name'], ascending=True, ignore_index=True, inplace=True)
    text_features = features.loc[features['datacolumn'] == 'textvalue'].copy()
    numeric_features = features.loc[features['datacolumn'] == 'numericvalue'].copy()
    assert features.shape[0] == text_features.shape[0] + numeric_features.shape[0]

    inputs['stayid'] = inputs['stayid'].astype('int')
    text_outputs = inputs.loc[inputs['featurename'].isin(text_features['name']),
                              ['stayid', 'featurename', 'textvalue']]\
        .pivot(index='stayid', columns='featurename', values='textvalue')
    numeric_outputs = inputs.loc[inputs['featurename'].isin(numeric_features['name']),
                                 ['stayid', 'featurename', 'numericvalue']]\
        .pivot(index='stayid', columns='featurename', values='numericvalue')
    assert features.shape[0] == text_outputs.shape[1] + numeric_outputs.shape[1]
    return numeric_outputs.join(text_outputs, how='inner')


if __name__ == '__main__':
    main()
