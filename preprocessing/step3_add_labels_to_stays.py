"""
Script to generate labels for three days ICU readmission for each ICU stay that updates the tables accordingly.
"""
import argparse
import datetime as dt
import sys

import pandas as pd

from preprocessing.step2_create_icu_stays import det_discharge_target_in_pdms
from preprocessing.variables import icus, im_icus
from research_database.research_database_communication import ResearchDBConnection


def main():
    parser = argparse.ArgumentParser(description='Generate labels for ICU stays.')
    parser.add_argument('password', type=str, help='Database password.')
    args, _ = parser.parse_known_args()

    db_conn = ResearchDBConnection(args.password)
    stays = db_conn.read_table_from_db('stays')

    # Process sorted stays for each patient, check if they should be used for prediction, and generate according label.
    # Logging variables for ICU stay creation.
    readmission_interval = dt.timedelta(days=3)
    stays['label'] = False
    # Note all stays refer to transfers at ICUs only, so no need to filter for that.
    # next_trans_* contain information about next transfers of same patient regardless of a new case.
    stays.loc[(stays['next_trans_intvon'] - stays['intbis'] <= readmission_interval), 'label'] = True
    # Create control statistics per year.
    num_transfers_per_year = {}
    for year, yearly_transfers in tuple(stays.loc[(stays['next_trans_intvon'] - stays['intbis'] <=
                                                   readmission_interval)].groupby(stays['ukmbis'].dt.year)):
        num_transfers_per_year[year] = yearly_transfers.shape[0]
    # Create control statistics per station.
    icu_readmissions = 0
    imc_readmission = 0
    for idx, _ in stays.loc[stays['label']].iterrows():
        unit = stays.loc[idx, 'next_trans_station']
        icu_readmissions += unit in icus  and unit not in im_icus
        imc_readmission += unit in im_icus
    print(f"Labeled {stays.loc[stays['label']].shape[0]} ({icu_readmissions}/{imc_readmission}) readmission as true.")
    print('(', end='')
    print(', '.join([f"{str(int(key))[2:4]}: {value} " for key, value in num_transfers_per_year.items()]), end=')\n')

    # Verify that readmissions only considered for ICUs.
    assert stays.loc[stays['label']].shape[0] == \
           stays.loc[(stays['next_trans_intvon'] - stays['intbis'] <= readmission_interval) &
                     (stays['next_trans_station'].isin(icus))].shape[0]

    assert stays.loc[(stays['intbis'] == stays['ukmbis']) & (stays['discharge'] == 'M')].shape[0] == 0
    stays.loc[(stays['intbis'] + readmission_interval >= stays['ukmbis']) & (stays['discharge'] == 'M'), 'label'] = True
    num_transfers_per_year = {}
    for year, yearly_transfers in tuple(stays.loc[(stays['intbis'] + readmission_interval >= stays['ukmbis']) &
                                                  (stays['discharge'] == 'M')].groupby(stays['ukmbis'].dt.year)):
        num_transfers_per_year[year] = yearly_transfers.shape[0]
    print(f"Labeled {stays.loc[(stays['intbis'] + readmission_interval >= stays['ukmbis']) & (stays['discharge'] == 'M')].shape[0]} "
          f"deaths at normal station as true")
    print('(', end='')
    print(', '.join([f"{str(int(key))[2:4]}: {value} " for key, value in num_transfers_per_year.items()]), end=')\n\n')

    print(f"Labeled {stays.loc[stays['label']].shape[0]} stays as true.")

    # Sample control stays for each label. Stratified by four stations and ending with death.
    # transfers = db_conn.read_table_from_db('transfers')
    # for icu in included_stations:
    #     format_control_samples(
    #         db_conn, transfers,
    #         stays.loc[stays['label'] & (stays['discharge'] != 'M') & (stays['station'].str.endswith(icu))].sample(4))
    # format_control_samples(db_conn, transfers, stays.loc[stays['label'] & (stays['discharge'] == 'M')].sample(4))
    # for icu in included_stations:
    #     format_control_samples(db_conn, transfers,
    #                            stays.loc[(~stays['label']) & (stays['station'].str.endswith(icu))].sample(5))

    # Update stays table with labels.
    db_conn.write_stays_to_db(stays)


def format_control_samples(db_conn, transfers, samples):
    case_number = 0
    for idx, _ in samples.iterrows():
        case_number += 1
        case_id = samples.at[idx, 'fallnummer']
        icu_start = samples.at[idx, 'intvon']
        icu_end = samples.at[idx, 'intbis']
        discharge_target = det_discharge_target_in_pdms(db_conn, case_id, icu_start, icu_end)
        print(f"case {case_number};Fallnummer: {samples.at[idx, 'fallnummer']} (patientid: {samples.at[idx, 'patientid']})\n"
              f"({samples.at[idx, 'label']});UKM: {samples.at[idx, 'ukmvon'].strftime('%d.%m.%Y %H:%M:%S')} - {samples.at[idx, 'ukmbis'].strftime('%d.%m.%Y %H:%M:%S')}\n"
              f";Stationen: {samples.at[idx, 'station'].replace(';', ',')}\n"
              f";Aufnahme: {samples.at[idx, 'intvon'].strftime('%d.%m.%Y %H:%M:%S')}\n"
              f";Entlassung: {samples.at[idx, 'intbis'].strftime('%d.%m.%Y %H:%M:%S')}\n"
              f";QS discharge: {discharge_target}")
        if pd.isnull(samples.at[idx, 'next_trans_intvon']):
            print(f";Station Wiederaufnahme: -\n"
                  f";Intervall: -")
        else:
            print(f";Aufnahme {samples.at[idx, 'next_trans_station']}: {samples.at[idx, 'next_trans_intvon'].strftime('%d.%m.%Y %H:%M:%S')}\n"
                  f";Intervall: {samples.at[idx, 'next_trans_intvon'] - samples.at[idx, 'intbis']}")
        print(";;;")

        patient_transfers = transfers.loc[transfers['patientid'] == samples.at[idx, 'patientid']].copy()
        transfers.sort_values(['fallnummer', 'intvon'], ascending=[True, True], inplace=True, ignore_index=True)
        patient_transfers.drop(columns=['id'], inplace=True)
        patient_transfers.to_csv(sys.stdout, sep=';')
        print(';;;')
        print(';;;')


if __name__ == '__main__':
    main()
