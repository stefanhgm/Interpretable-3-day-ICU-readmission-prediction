import argparse
import os

import pandas as pd

from data_description.stays_description_table import add_mean_sd_to_table_td, dict_to_string, add_mean_sd_to_table
from research_database.research_database_communication import ResearchDBConnection


def main():
    parser = argparse.ArgumentParser(description='Create table to summarize cohort characteristics.')
    parser.add_argument('password', type=str, help='Database password.')
    parser.add_argument('--data', type=str, help='CSV file containing features and targets.')
    parser.add_argument('--output_path', type=str, help='Directory to store generated plots.')
    args, _ = parser.parse_known_args()

    db_conn = ResearchDBConnection(args.password)
    stays = db_conn.read_table_from_db('mimic_stays')

    # Add additional fields to stays.
    stays['icu_duration'] = stays['icu_outtime'] - stays['icu_intime']

    stays_false = stays.loc[~stays['label']]
    stays_true = stays.loc[stays['label']]

    # Create table for cohort characteristics.
    table = pd.DataFrame({'Characteristic': [], 'All stays': [], 'Labeled false': [], 'Labeled true': []})
    table = table.append({'Characteristic': 'Number of stays',
                          'All stays': f"{stays.shape[0]} ({stays.shape[0]/stays.shape[0]:.1%})",
                          'Labeled false': f"{stays_false.shape[0]} ({stays_false.shape[0]/stays.shape[0]:.1%})",
                          'Labeled true': f"{stays_true.shape[0]} ({stays_true.shape[0]/stays.shape[0]:.1%})"},
                         ignore_index=True)

    # Patients.
    table = table.append({'Characteristic': 'Number of patients',
                          'All stays': f"{stays['hadm_id'].unique().shape[0]} "
                                f"({stays['hadm_id'].unique().shape[0]/stays['hadm_id'].unique().shape[0]:.1%})",
                          'Labeled false': f"{stays_false['hadm_id'].unique().shape[0]} "
                                f"({stays_false['hadm_id'].unique().shape[0]/stays['hadm_id'].unique().shape[0]:.1%})",
                          'Labeled true': f"{stays_true['hadm_id'].unique().shape[0]} "
                                f"({stays_true['hadm_id'].unique().shape[0]/stays['hadm_id'].unique().shape[0]:.1%})"},
                         ignore_index=True)

    # Age
    age = 'admission_age'
    table = add_mean_sd_to_table(table, stays, stays_false, stays_true, age, 'Mean age (SD)')

    # Gender
    gender = 'gender'
    table = table.append({'Characteristic': 'Gender',
                          'All stays': f"{dict_to_string(stays[gender].value_counts().to_dict())}",
                          'Labeled false': f"{dict_to_string(stays_false[gender].value_counts().to_dict())}",
                          'Labeled true': f"{dict_to_string(stays_true[gender].value_counts().to_dict())}"},
                         ignore_index=True)

    # Station
    station = 'last_careunit'
    table = table.append({'Characteristic': 'Stations',
                          'All stays': f"{dict_to_string(stays[station].value_counts().to_dict())}",
                          'Labeled false': f"{dict_to_string(stays_false[station].value_counts().to_dict())}",
                          'Labeled true': f"{dict_to_string(stays_true[station].value_counts().to_dict())}"},
                         ignore_index=True)

    # Length of stay
    length_of_stay = 'icu_duration'
    table =\
        add_mean_sd_to_table_td(table, stays, stays_false, stays_true, length_of_stay, 'Mean length of ICU stay (SD)')

    table.to_csv(os.path.join(args.output_path, 'cohort_description_mimic.csv'), sep=',', header=True, index=False)


if __name__ == '__main__':
    main()
