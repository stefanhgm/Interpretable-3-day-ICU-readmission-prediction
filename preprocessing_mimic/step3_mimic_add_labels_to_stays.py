"""
This script generates labels for the mimic data.
"""
import argparse
import datetime as dt

from preprocessing_mimic.variables import m_icus, m_normal_stations, m_im_icus
from research_database.research_database_communication import ResearchDBConnection, MIMICDBConnection


def main():
    parser = argparse.ArgumentParser(description='Generate labels for mimic ICU stays.')
    parser.add_argument('password', type=str, help='Database password.')
    args, _ = parser.parse_known_args()

    db_conn = ResearchDBConnection(args.password)
    mimic_stays = db_conn.read_table_from_db('mimic_stays')
    # Use mimic concept icustay_details to derive next icu stay. Should be cleaner than raw transfers.
    mimic_db_conn = MIMICDBConnection(args.password)
    mimic_transfers = mimic_db_conn.read_table_from_db('mimic_core', 'transfers')
    mimic_icu_stays = mimic_db_conn.read_table_from_db('mimic_icu', 'icustays')

    readmission_interval = dt.timedelta(days=3)
    mimic_stays['label'] = False
    # Necessary to iterate over all stays because in contrast to original cohort no info about next station.
    # Also more complex case distinction because normal stations included in transfers (original cohort only ICU/IMC).
    icu_imc_transfers_gte_1h =\
        mimic_transfers.loc[(((mimic_transfers['careunit'].isin(m_icus)) &
                            mimic_transfers['transfer_id'].isin(mimic_icu_stays['stay_id'])) |
                             (mimic_transfers['careunit'].isin(m_im_icus))) &
                            ((mimic_transfers['outtime'] - mimic_transfers['intime']) >= dt.timedelta(hours=1))]
    normal_transfers_gte_1h =\
        mimic_transfers.loc[(mimic_transfers['careunit'].isin(m_normal_stations)) &
                            ((mimic_transfers['outtime'] - mimic_transfers['intime']) >= dt.timedelta(hours=1))]
    icu_readmissions = 0
    imc_readmission = 0
    for idx, _ in mimic_stays.iterrows():
        # First, identify earliest admission to any ICU or IMC within relevant interval with a minimum distance of 1h.
        icu_imc_transfers_within_interval =\
            icu_imc_transfers_gte_1h.loc[
                (mimic_stays.loc[idx, 'subject_id'] == icu_imc_transfers_gte_1h['subject_id']) &
                (mimic_stays.loc[idx, 'hadm_id'] == icu_imc_transfers_gte_1h['hadm_id']) &
                ((mimic_stays.loc[idx, 'icu_outtime'] + dt.timedelta(hours=1)) <= icu_imc_transfers_gte_1h['intime']) &
                ((mimic_stays.loc[idx, 'icu_outtime'] + readmission_interval) >= icu_imc_transfers_gte_1h['intime'])]
        if icu_imc_transfers_within_interval.shape[0] == 0:
            continue
        # Second, check if before earliest ICU/IMC admission at a normal ward station, so a relevant readmission.
        earliest_icu_imc_readmission_time = min(icu_imc_transfers_within_interval['intime'])
        # Check if before ICU readmission at a normal ward station.
        transfer_to_normal_before_icu_readmission =\
            normal_transfers_gte_1h.loc[(mimic_stays.loc[idx, 'subject_id'] == normal_transfers_gte_1h['subject_id']) &
                                        (mimic_stays.loc[idx, 'hadm_id'] == normal_transfers_gte_1h['hadm_id']) &
                                        (mimic_stays.loc[idx, 'icu_outtime'] <= normal_transfers_gte_1h['intime']) &
                                        (earliest_icu_imc_readmission_time >= normal_transfers_gte_1h['outtime'])]
        if transfer_to_normal_before_icu_readmission.shape[0] > 0:
            mimic_stays.loc[idx, 'label'] = True
            unit =\
                icu_imc_transfers_within_interval.loc[icu_imc_transfers_within_interval['intime'].idxmin(), 'careunit']
            icu_readmissions += unit in m_icus
            imc_readmission += unit in m_im_icus
    print(f"Labeled {mimic_stays.loc[mimic_stays['label']].shape[0]} ({icu_readmissions}/{imc_readmission})"
          f" ICU readmission after ward within 3 days as true.")

    # Label death within 3 days.
    normal_transfers_gte_1h =\
        mimic_transfers.loc[(mimic_transfers['careunit'].isin(m_normal_stations)) &
                            ((mimic_transfers['outtime'] - mimic_transfers['intime']) >= dt.timedelta(hours=1))]
    death_events = 0
    for idx, _ in\
        mimic_stays.loc[(~(mimic_stays['label'])) & (~(mimic_stays['deathtime'].isna())) &
                        ((mimic_stays['deathtime'] - mimic_stays['icu_outtime']) <= readmission_interval)].iterrows():
        # Check if before death admission at a normal ward station, so a relevant death event.
        transfer_to_normal_before_death =\
            normal_transfers_gte_1h.loc[(mimic_stays.loc[idx, 'subject_id'] == normal_transfers_gte_1h['subject_id']) &
                                        (mimic_stays.loc[idx, 'hadm_id'] == normal_transfers_gte_1h['hadm_id']) &
                                        (mimic_stays.loc[idx, 'icu_outtime'] <= normal_transfers_gte_1h['intime']) &
                                        (mimic_stays.loc[idx, 'deathtime'] > normal_transfers_gte_1h['intime'])]
        if transfer_to_normal_before_death.shape[0] > 0:
            mimic_stays.loc[idx, 'label'] = True
            death_events += 1
    print(f"Labeled {death_events} death events after discharge to normal station within 3 days as true.")
    print(f"Labeled {mimic_stays.loc[mimic_stays['label']].shape[0]} ICU readmission and death within 3 days as true.")

    # Update stays table with labels.
    db_conn.write_table_to_db('mimic_stays', mimic_stays)


if __name__ == '__main__':
    main()
