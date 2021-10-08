"""
Script to select stays in the mimic-iv database for external evaluation.
"""
import argparse
import collections

import datetime as dt
import pandas as pd
from preprocessing_mimic.variables import m_included_stations, m_icus, m_im_icus
from research_database.research_database_communication import ResearchDBConnection, MIMICDBConnection

# Global variable to keep track of number of mimic_stays per year to show excluded cases per year.
num_mimic_stays_per_year = {}


def print_total_mimic_stays(mimic_stays):
    global num_mimic_stays_per_year
    old_num_mimic_stays_per_year = num_mimic_stays_per_year
    num_mimic_stays_per_year = {}
    for year, yearly_mimic_stays in tuple(mimic_stays.groupby(mimic_stays['dischtime'].dt.year)):
        # Only use last digit of year for rough overview.
        num_mimic_stays_per_year[int(str(year)[3:4])] =\
            num_mimic_stays_per_year.get(int(str(year)[3:4]), 0) + yearly_mimic_stays.shape[0]
    num_mimic_stays_per_year = collections.OrderedDict(sorted(num_mimic_stays_per_year.items()))
    print(f"Total mimic_stays: {mimic_stays.shape[0]}", end=' ')
    print('(', end='')
    print(', '.join([f"{key}: {value} [{value - old_num_mimic_stays_per_year.get(key, 0)}]"
                     for key, value in num_mimic_stays_per_year.items()]), end=')\n\n')


def main():
    parser = argparse.ArgumentParser(description='Convert mimic-iv derived icu stays into stays.')
    parser.add_argument('password', type=str, help='Database password.')
    args, _ = parser.parse_known_args()

    # Define important variables for stay exclusion.
    min_observation_period = dt.timedelta(hours=72)

    print("Read mimic derived icu stays table and write them into designated mimic_stays table.")
    db_conn = MIMICDBConnection(args.password)
    mimic_stays = db_conn.read_table_from_db('mimic_derived', 'icustay_detail')
    original_num_mimic_stays = mimic_stays.shape[0]
    original_num_patients = len(set(mimic_stays['subject_id'].to_list()))
    # Add first and last care unit.
    mimic_stations = \
        db_conn.read_table_from_db('mimic_icu', 'icustays')[['stay_id', 'first_careunit', 'last_careunit']]
    mimic_stays = mimic_stays.merge(mimic_stations, on='stay_id', suffixes=('', '_y'))
    # Add death time.
    mimic_admission = \
        db_conn.read_table_from_db('mimic_core', 'admissions')
    mimic_admission = mimic_admission.loc[(mimic_admission['discharge_location'] == 'DIED') &
                                          (mimic_admission['deathtime'].notnull())]
    # remove single unnecessary duplicate.
    mimic_admission.drop(mimic_admission[(mimic_admission['subject_id'] == 13771243) &
                                         (mimic_admission['hadm_id'] == 20491701)].index, inplace=True)
    mimic_stays = mimic_stays.merge(mimic_admission[['subject_id', 'deathtime']], how='left', on='subject_id')
    assert mimic_stays.shape[0] == original_num_mimic_stays
    print_total_mimic_stays(mimic_stays)

    # Verify that only adult patients.
    assert all(mimic_stays['admission_age'] >= 18)

    # Global changes to recordings as in step1_global_changes_to_research_database.py not performed for MIMIC-IV
    # because we assume that data already properly filtered.
    # Exclude ICU stays of children.
    old_num_stays = mimic_stays.shape[0]
    mimic_stays.drop(mimic_stays[mimic_stays['admission_age'] < 18].index, inplace=True)
    print(f"Excluded stays of children (n={old_num_stays - mimic_stays.shape[0]})")

    print("2. Repair, if possible, or remove erroneous derived mimic icu stays.")
    print("2a. Remove icu stays with discharge after admission.")
    bad_patients = set(mimic_stays.loc[mimic_stays['admittime'] >= mimic_stays['dischtime'], 'subject_id'].to_list())
    old_num_stays = mimic_stays.shape[0]
    mimic_stays.drop(mimic_stays[mimic_stays['subject_id'].isin(bad_patients)].index, inplace=True)
    print(f"\tRemoved {len(bad_patients)} patients with {old_num_stays - mimic_stays.shape[0]} mimic_stays.")

    # Not relevant for MIMIC.
    # "2b. Repair hospital discharge date 2099-12-31 00:00:00 and location when correct data exists for the case."
    # "2c. Remove remaining mimic_stays with hospital discharge 2099-12-31 00:00:00, since no correct discharge "
    #      "information can be obtained."
    # "2d. Verify that all cases have the same hospital admission and discharge dates and targets."

    print("2e. Remove patients with overlapping hospital mimic_stays. Unambiguous repair impossible.")
    mimic_stays.sort_values(['subject_id', 'admittime', 'dischtime'], ascending=[True, False, False], inplace=True,
                            ignore_index=True)
    bad_patients = set()
    next_admission_idx = -1
    for idx, _ in mimic_stays.iterrows():
        if next_admission_idx != -1 and \
                mimic_stays.at[idx, 'subject_id'] == mimic_stays.at[next_admission_idx, 'subject_id'] and \
                mimic_stays.at[idx, 'admittime'] < mimic_stays.at[next_admission_idx, 'admittime'] \
                < mimic_stays.at[idx, 'dischtime']:
            bad_patients.add(mimic_stays.at[idx, 'subject_id'])
        next_admission_idx = idx
    old_num_stays = mimic_stays.shape[0]
    mimic_stays.drop(mimic_stays[mimic_stays['subject_id'].isin(bad_patients)].index, inplace=True)
    print(f"\tRemoved {len(bad_patients)} patients with {old_num_stays - mimic_stays.shape[0]} mimic_stays.")

    # Not relevant for MIMIC.
    # "2f. Remove patient that was not distinguishable during label review."

    print("3. Repair, if possible, or remove erroneous hospital mimic_stays (icu_intime, icu_outtime).")

    print("3a. Remove mimic_stays with admission after discharge.")
    bad_patients = set(mimic_stays.loc[mimic_stays['icu_intime'] >= mimic_stays['icu_outtime'], 'subject_id'].to_list())
    old_num_mimic_stays = mimic_stays.shape[0]
    mimic_stays.drop(mimic_stays[mimic_stays['subject_id'].isin(bad_patients)].index, inplace=True)
    print(f"\tRemoved {len(bad_patients)} patients with {old_num_mimic_stays - mimic_stays.shape[0]} mimic_stays.")

    # Not relevant for MIMIC.
    # In MIMIC data ICU discharge might happen after hospital discharge.
    # "3b. Verify that all mimic_stays within their hospital admission."

    print("3c. Remove patients with overlapping stays. Repair might be possible, but only few so safer to remove.")
    mimic_stays.sort_values(['subject_id', 'icu_intime', 'icu_outtime'], ascending=[True, False, False], inplace=True,
                            ignore_index=True)
    while True:
        # Repeat procedure until no problematic transfer detected. There might be nested problems.
        # Deal with all cases of overlapping mimic_stays.
        patient_id = -1
        next_transfer_idx = -1
        bad_patients = set()
        num_completely = 0
        num_same_start = 0
        num_diff_start = 0
        for idx, _ in mimic_stays.iterrows():
            if mimic_stays.at[idx, 'subject_id'] != patient_id:
                # New patient encountered.
                patient_id = mimic_stays.at[idx, 'subject_id']
            else:
                # Reset patient id to ensure that surrounding of this entry non edited twice.
                patient_id = -1
                # Case 1: Earlier stay completely contains following one
                # Next stay:   |    |
                # Prev stay: |        |
                if mimic_stays.at[idx, 'icu_intime'] < mimic_stays.at[next_transfer_idx, 'icu_intime'] and \
                        mimic_stays.at[idx, 'icu_outtime'] > mimic_stays.at[next_transfer_idx, 'icu_outtime']:
                    num_completely += 1
                    bad_patients.add(mimic_stays.at[idx, 'subject_id'])
                # Case 2: Same start
                # Next stay:  |     |    /  |     |
                # Prev stay:  |       |  /  |     |
                elif mimic_stays.at[idx, 'icu_intime'] == mimic_stays.at[next_transfer_idx, 'icu_intime']:
                    same_end = mimic_stays.at[idx, 'icu_outtime'] == mimic_stays.at[next_transfer_idx, 'icu_outtime']
                    num_same_start += 1
                    bad_patients.add(mimic_stays.at[idx, 'subject_id'])
                # Case 3: Different start
                # Next stay:    |     |  /    |   |
                # Prev stay:  |     |    /  |     |
                elif mimic_stays.at[idx, 'icu_intime'] < mimic_stays.at[next_transfer_idx, 'icu_intime'] \
                        < mimic_stays.at[idx, 'icu_outtime']:
                    same_end = mimic_stays.at[idx, 'icu_outtime'] == mimic_stays.at[next_transfer_idx, 'icu_outtime']
                    num_diff_start += 1
                    bad_patients.add(mimic_stays.at[idx, 'subject_id'])
                    # Earlier: Picked longer entry.
            next_transfer_idx = idx
        if len(bad_patients) == 0:
            break
    old_num_mimic_stays = mimic_stays.shape[0]
    mimic_stays.drop(mimic_stays[mimic_stays['subject_id'].isin(bad_patients)].index, inplace=True)
    print(f"\tRemoved {len(bad_patients)} patients with {old_num_mimic_stays - mimic_stays.shape[0]} mimic_stays.")
    print(f"[Types: {num_completely} completely, {num_same_start} same start, {num_diff_start} diff start]")

    print("3d. Remove mimic_stays with death before ICU stay.")
    bad_patients = set(mimic_stays.loc[mimic_stays['deathtime'] < mimic_stays['icu_intime'], 'subject_id'].to_list())
    old_num_mimic_stays = mimic_stays.shape[0]
    mimic_stays.drop(mimic_stays[mimic_stays['subject_id'].isin(bad_patients)].index, inplace=True)
    print(f"\tRemoved {len(bad_patients)} patients with {old_num_mimic_stays - mimic_stays.shape[0]} mimic_stays.")
    print_total_mimic_stays(mimic_stays)

    print("")
    print(f"Perform stay selection analogously to cohort of ANIT-UKM")

    # Exclude all stays that are not managed by "Anesthesia, Critical Care and Pain Medicine"
    old_num_mimic_stays = mimic_stays.shape[0]
    mimic_stays.drop(mimic_stays[~(mimic_stays['last_careunit'].isin(m_included_stations))].index, inplace=True)
    print(f"Excluded mimic_stays without last icu included (n={old_num_mimic_stays - mimic_stays.shape[0]})")
    print_total_mimic_stays(mimic_stays)

    # Exclude mimic_stays with death at ICU.
    old_num_mimic_stays = mimic_stays.shape[0]
    mimic_stays.drop(mimic_stays[(mimic_stays['deathtime'].notnull()) &
                                 (mimic_stays['deathtime'] >= mimic_stays['icu_intime']) &
                                 (mimic_stays['deathtime'] <= mimic_stays['icu_outtime'])].index, inplace=True)
    assert mimic_stays.loc[(mimic_stays['deathtime'].notnull()) &
                           (mimic_stays['deathtime'] < mimic_stays['icu_intime'])].shape[0] == 0
    print(f"Excluded mimic_stays with death at ICU (n={old_num_mimic_stays - mimic_stays.shape[0]})")

    # Check if patient admitted to an ICU after death. Indicates that probably died on ICU.
    # Already removed death time during ICU stay and asserted that not before ICU stay.
    # Now consider three days afterwards because this is relevant interval.
    old_num_mimic_stays = mimic_stays.shape[0]
    mimic_transfers = db_conn.read_table_from_db('mimic_core', 'transfers')
    bad_stays = []
    for idx, _ in mimic_stays.loc[(mimic_stays['deathtime'].notnull()) &
                                  (mimic_stays['deathtime'] <
                                   (mimic_stays['icu_outtime'] + dt.timedelta(days=3)))].iterrows():
        icu_transfers_after_death =\
            mimic_transfers.loc[(mimic_transfers['subject_id'] == mimic_stays.loc[idx, 'subject_id']) &
                                (mimic_transfers['intime'] > mimic_stays.loc[idx, 'deathtime']) &
                                (mimic_transfers['careunit'].isin(m_icus) | mimic_transfers['careunit'].isin(m_im_icus))]
        if icu_transfers_after_death.shape[0] > 0:
            bad_stays.append(mimic_stays.loc[idx, 'stay_id'])
    mimic_stays.drop(mimic_stays[mimic_stays['stay_id'].isin(bad_stays)].index, inplace=True)
    print(f"Excluded mimic_stays ICU/IMC admission after death (n={old_num_mimic_stays - mimic_stays.shape[0]})")
    print_total_mimic_stays(mimic_stays)

    # Previously also excluded death after ICU discharge within 12h. However, analysis showed that usually transferred
    # to normal station and it is unclear whether due to palliative care or not. So, keep all deaths after ICU stay.

    # Exclude mimic_stays with insufficient observation period.
    old_num_mimic_stays = mimic_stays.shape[0]
    mimic_stays.drop(
        mimic_stays[(~(mimic_stays['deathtime'] <= mimic_stays['icu_outtime'] + min_observation_period)) &
                    (mimic_stays['icu_outtime'] + min_observation_period > mimic_stays['dischtime'])].index,
        inplace=True)
    print(f"Excluded mimic_stays without min observation interval of {min_observation_period} after ICU stay "
          f"(n={old_num_mimic_stays - mimic_stays.shape[0]})")
    deaths_outside_icu_3d =\
        mimic_stays.loc[(mimic_stays['deathtime'] <= mimic_stays['icu_outtime'] + min_observation_period)].shape[0]
    print(f"(Deaths outside ICU within 3d n={deaths_outside_icu_3d})")
    # All mimic_stays that are observed less than 3d after ICU die at a normal station.
    print_total_mimic_stays(mimic_stays)

    # Remove implausible transfers. Might add further filters here.
    # Remove cases based on HF chart events.
    old_num_mimic_stays = mimic_stays.shape[0]
    icu_stays_hf_min_max = db_conn.read_icu_stays_with_min_max_hf_from_db()
    # Delete all transfers where the transfer itself and all transfers merged into it contain no HF measurement.
    mimic_stays.drop(mimic_stays[~(mimic_stays['stay_id'].isin(icu_stays_hf_min_max['stay_id']))].index, inplace=True)
    print(f"Excluded stays with no hf during stay (n={old_num_mimic_stays - mimic_stays.shape[0]})")
    # Add min and max hf recording times.
    old_num_mimic_stays = mimic_stays.shape[0]
    mimic_stays = mimic_stays.merge(icu_stays_hf_min_max[['stay_id', 'min', 'max']], how='left', on='stay_id')
    assert mimic_stays.shape[0] == old_num_mimic_stays
    # Remove all stays with less than 1h of hf recordings.
    mimic_stays.drop(mimic_stays[mimic_stays['max'] - mimic_stays['min'] < pd.Timedelta(hours=1)].index, inplace=True)
    print(f"Excluded stays with less than 1h hf rec. during stay (n={old_num_mimic_stays - mimic_stays.shape[0]})")
    # Remove all transfers with less than 2h of hf recordings.
    old_num_mimic_stays = mimic_stays.shape[0]
    mimic_stays.drop(mimic_stays[mimic_stays['max'] - mimic_stays['min'] < pd.Timedelta(hours=1)].index, inplace=True)
    print(f"Excluded stays with less than 2h hf rec. during stay (n={old_num_mimic_stays - mimic_stays.shape[0]})")
    mimic_stays.drop(columns=['min', 'max'], inplace=True)
    print_total_mimic_stays(mimic_stays)

    # Write resulting ICU stays to database.
    mimic_stays.sort_values(['subject_id'], ascending=[True], inplace=True, ignore_index=True)
    num_patients = len(set(mimic_stays['subject_id'].to_list()))
    num_mimic_stays = mimic_stays.shape[0]
    print(f"In total removed {original_num_patients - num_patients} of {original_num_patients} patients.")
    print(f"In total removed {original_num_mimic_stays - num_mimic_stays} of {original_num_mimic_stays} mimic_stays.")
    db_conn = ResearchDBConnection(args.password)
    db_conn.write_table_to_db('mimic_stays', mimic_stays)


if __name__ == '__main__':
    main()
