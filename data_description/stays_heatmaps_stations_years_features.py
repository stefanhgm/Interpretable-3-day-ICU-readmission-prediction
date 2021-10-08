import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing.step6_output_features_and_labels import pivot_inputs_for_features
from research_database.research_database_communication import ResearchDBConnection


def main(included_stations=None):
    parser = argparse.ArgumentParser(description='Create heatmap to visualize stays for years and stations.')
    parser.add_argument('password', type=str, help='Database password.')
    parser.add_argument('--output_path', type=str, help='Directory to store generated plots.')
    args, _ = parser.parse_known_args()

    db_conn = ResearchDBConnection(args.password)
    stays = db_conn.read_table_from_db('stays')
    features = db_conn.read_table_from_db('features')
    inputs = db_conn.read_table_from_db('inputs')

    # For debugging reduce number of relevant stays.
    # stays.drop(stays[stays['intvon'].dt.year > 2006].index, inplace=True)

    # Filter features if necessary.
    features = features.loc[features['name'].str.contains('SAPS')]

    # Add additional fields to stays.
    stays['icu_duration'] = stays['intbis'] - stays['intvon']
    stays['intbis_year'] = stays['intbis'].dt.year
    stays = stays.set_index('id')
    feature_output = pivot_inputs_for_features(features, inputs)
    first_feature_column = feature_output.columns.tolist()[0]
    # Set all fields to 0/1 indicating whether unknown value or not.
    feature_output = feature_output.applymap(lambda x: 0 if pd.isna(x) else 1)
    old_num_stays = stays.shape[0]
    stays = stays.join(feature_output, how='inner')
    assert old_num_stays == stays.shape[0]

    # Create heatmap for years and stations.
    # Inspired from https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
    years = stays['intbis_year'].unique().tolist()
    years.sort()
    stations = included_stations

    years_stations_heatmap = np.zeros((len(stations), len(years)), dtype=int)
    features_stations_heatmap = np.zeros((len(stations), len(features)), dtype=float)
    features_years_heatmap = np.zeros((len(years), len(features)), dtype=float)

    counts_years = [0 for _ in years]
    for idx, _ in stays.iterrows():
        year = stays.loc[idx, 'intbis_year']
        counts_years[years.index(year)] += 1
        features_years_heatmap[years.index(year), :] = features_years_heatmap[years.index(year), :] + \
            stays.loc[idx, first_feature_column:stays.columns.tolist()[-1]]
        stations_string = stays.loc[idx, 'station']
        for station in stations_string.split(';'):
            years_stations_heatmap[stations.index(station), years.index(year)] += 1
            features_stations_heatmap[stations.index(station), :] =\
                features_stations_heatmap[stations.index(station), :] + \
                stays.loc[idx, first_feature_column:stays.columns.tolist()[-1]]
    counts_multiple_years = np.sum(years_stations_heatmap, axis=0)
    counts_stations = np.sum(years_stations_heatmap, axis=1)
    features_stations_heatmap = features_stations_heatmap / counts_stations[:, np.newaxis]
    features_years_heatmap = features_years_heatmap / np.array(counts_years)[:, np.newaxis]

    station_labels = stations.copy()
    multiple_year_labels = years.copy()
    year_labels = years.copy()
    feature_labels = stays.columns.tolist()[stays.columns.tolist().index(first_feature_column):]
    # Add cumulative numbers to labels.
    for i in range(len(year_labels)):
        year_labels[i] = str(year_labels[i]) + f" (n={counts_years[i]:,})"
    for i in range(len(multiple_year_labels)):
        multiple_year_labels[i] = str(multiple_year_labels[i]) + f" (n={counts_multiple_years[i]:,})"
    for i in range(len(station_labels)):
        station_labels[i] = str(station_labels[i]) + f" (n={counts_stations[i]:,})"

    # Format years_stations_heatmap
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(years_stations_heatmap, interpolation='nearest')
    ax.set_xticks(np.arange(len(multiple_year_labels)))
    ax.set_yticks(np.arange(len(station_labels)))
    ax.set_xticklabels(multiple_year_labels)
    ax.set_yticklabels(station_labels)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(station_labels)):
        for j in range(len(multiple_year_labels)):
            text = ax.text(j, i, f"{years_stations_heatmap[i, j]:,}", ha="center", va="center", color="w")
    ax.set_title("Included stays over starting year and station (stations of merged stays are considered separately)")
    plt.savefig(os.path.join(args.output_path, 'years_stations_heatmap.png'), dpi=300, bbox_inches='tight')

    # Format features_stations_heatmap
    fig, ax = plt.subplots(figsize=(25, 7))
    im = ax.imshow(features_stations_heatmap, interpolation='nearest')
    ax.set_xticks(np.arange(len(feature_labels)))
    ax.set_yticks(np.arange(len(station_labels)))
    ax.set_xticklabels(feature_labels)
    ax.set_yticklabels(station_labels)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(station_labels)):
        for j in range(len(feature_labels)):
            text = ax.text(j, i, f"{features_stations_heatmap[i, j]:.2f}", ha="center", va="center", color="w")
    ax.set_title("Feature completeness for stations (stations of merged stays are considered separately)")
    plt.savefig(os.path.join(args.output_path, 'features_stations_heatmap.png'), dpi=300, bbox_inches='tight')

    # Format features_years_heatmap
    fig, ax = plt.subplots(figsize=(25, 10))
    im = ax.imshow(features_years_heatmap, interpolation='nearest')
    ax.set_xticks(np.arange(len(feature_labels)))
    ax.set_yticks(np.arange(len(year_labels)))
    ax.set_xticklabels(feature_labels)
    ax.set_yticklabels(year_labels)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(year_labels)):
        for j in range(len(feature_labels)):
            text = ax.text(j, i, f"{features_years_heatmap[i, j]:.2f}", ha="center", va="center", color="w")
    ax.set_title("Feature completeness for years")
    plt.savefig(os.path.join(args.output_path, 'features_years_heatmap.png'), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
