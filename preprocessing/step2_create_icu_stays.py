"""
Script to convert the raw information about patient transfers and discharges obtained from the HIS into ICU stays.
"""
import argparse

import collections
import numpy as np
import pandas as pd
import datetime as dt

from helper.util import output_wrapper
from preprocessing.variables import station_mapping, stations, included_stations, anesthesiology_stations
from research_database.research_database_communication import ResearchDBConnection

# Global variable to keep track of number of transfers per year to show excluded cases per year.
num_transfers_per_year = {}


def print_total_transfers(transfers):
    global num_transfers_per_year
    old_num_transfers_per_year = num_transfers_per_year
    num_transfers_per_year = {}
    for year, yearly_transfers in tuple(transfers.groupby(transfers['ukmbis'].dt.year)):
        num_transfers_per_year[year] = yearly_transfers.shape[0]
    num_transfers_per_year = collections.OrderedDict(sorted(num_transfers_per_year.items()))
    print(f"Total transfers: {transfers.shape[0]}", end=' ')
    print('(', end='')
    print(', '.join([f"{str(int(key))[2:4]}: {value} [{value - old_num_transfers_per_year.get(key, 0)}]"
                     for key, value in num_transfers_per_year.items()]), end=')\n\n')


@output_wrapper
def parallel_stay_hf_check(password, transfer_id, transfers):
    db_conn = ResearchDBConnection(password)
    hf_values = db_conn.read_values_from_db('Heart rate', 'continuous', case_id=transfer_id)
    empty_stays_idx = []
    for idx in transfers.index:
        transfer_start, transfer_end = transfers.loc[idx, ['intvon', 'intbis']]
        hf_case = hf_values.loc[(hf_values['displaytime'] >= transfer_start) &
                                (hf_values['displaytime'] <= transfer_end)]
        print(idx, transfer_id, transfers.loc[idx, 'patientid'],  hf_case.shape[0])
        if hf_case.shape[0] == 0:
            empty_stays_idx.append(idx)
    return empty_stays_idx


def is_in_stations(transfers, relevant_stations):
    return transfers['station'].isin(relevant_stations)


def is_in_obs_interval(transfers, obs_interval):
    return (transfers['intvon'].dt.year >= obs_interval[0]) & (transfers['intbis'].dt.year <= obs_interval[1])


def is_in_pdms(transfers, start_times_pdms):
    return ((transfers['station'] == '19A OST') & (transfers['intvon'] >= start_times_pdms['19A OST'])) | \
           ((transfers['station'] == '19B OST') & (transfers['intvon'] >= start_times_pdms['19B OST'])) | \
           ((transfers['station'] == 'ANAES INT 2') & (transfers['intvon'] >= start_times_pdms['ANAES INT 2'])) | \
           ((transfers['station'] == 'ANAES PAS') & (transfers['intvon'] >= start_times_pdms['ANAES PAS']))


def next_trans_is_in_pdms(transfers, start_times_pdms):
    return ((transfers['next_trans_station'] == '19A OST') &
            (transfers['next_trans_intvon'] >= start_times_pdms['19A OST'])) | \
        ((transfers['next_trans_station'] == '19B OST') &
         (transfers['next_trans_intvon'] >= start_times_pdms['19B OST'])) | \
        ((transfers['next_trans_station'] == 'ANAES INT 2') &
         (transfers['next_trans_intvon'] >= start_times_pdms['ANAES INT 2'])) | \
        ((transfers['next_trans_station'] == 'ANAES PAS') &
         (transfers['next_trans_intvon'] >= start_times_pdms['ANAES PAS']))


def is_not_excl_speciality_on_19b(transfers, excl_speciality_at_19b):
    return (transfers['station'] != '19B OST') | (~transfers['speciality'].isin(excl_speciality_at_19b))


def select_subset_transported(subset_transfers, transported, offset_transport):
    if subset_transfers.shape[0] > 0:
        subset_transfers = subset_transfers.loc[subset_transfers.apply(
            lambda transfer:(transported[
                             (transported['patientid'] == transfer['fallnummer']) &
                             (transported['displaytime'] + offset_transport >= transfer['intbis']) &
                             (transported['displaytime'] <= transfer['next_trans_intvon'])].shape[0] > 0), axis=1)]
    return subset_transfers


def det_max_hf_4h_around_discharge(db_conn, case_id, icu_end, next_icu_start):
    # Determine the maximum interval around the discharge in which hf measurements are missing.
    hf_during_discharge = \
        db_conn.read_records_from_db(['numericvalue'], 34, case_id,
                                     icu_end - pd.Timedelta(hours=4), next_icu_start + pd.Timedelta(hours=4))
    hf_during_discharge = hf_during_discharge.loc[hf_during_discharge['numericvalue'] != '0']
    hf_during_discharge.sort_values(['displaytime'], ascending=[True], inplace=True, ignore_index=True)
    hf_during_discharge['delta'] = \
        (hf_during_discharge['displaytime'] - hf_during_discharge['displaytime'].shift())
    if hf_during_discharge.shape[0] >= 2:
        max_interval_no_hf = hf_during_discharge['delta'].max()
    else:
        max_interval_no_hf = pd.Timedelta(hours=8)

    return max_interval_no_hf


def det_discharge_target_in_pdms(db_conn, case_id, icu_start, icu_end):
    # Determine the discharge target as inserted into the ICU PDMS system.
    discharge_target = \
        db_conn.read_records_from_db(['numericvalue_text'], 4024, case_id, icu_start, icu_end + pd.Timedelta(hours=2))
    if discharge_target.shape[0] > 0:
        discharge_target = discharge_target.iloc[-1]['numericvalue_text']
    else:
        discharge_target = 'empty'

    return discharge_target


def det_discharge_time_in_pdms(db_conn, case_id, icu_start, icu_end):
    # Determine the time the discharge was entered into the ICU PDMS system.
    discharge_time = \
        db_conn.read_records_from_db(['displaytime'], 4024, case_id, icu_start, icu_end + pd.Timedelta(hours=2))
    if discharge_time.shape[0] > 0:
        discharge_time = discharge_time.iloc[-1, 3]
    else:
        discharge_time = dt.datetime(2099, 1, 1, hour=0, minute=0, second=0)

    return discharge_time


def main():
    parser = argparse.ArgumentParser(description='Convert transfers into stays.')
    parser.add_argument('password', type=str, help='Database password.')
    args, _ = parser.parse_known_args()
    print("Convert transfers into stays.")

    db_conn = ResearchDBConnection(args.password)
    transfers = db_conn.read_table_from_db('transfers')

    # Change incorrect dates indicating future.
    transfers['intbis'] = transfers['intbis'].astype('str')
    transfers['intbis'] = transfers['intbis'].str.replace('4000-12-31 00:00:00', '2021-12-31 00:00:00', regex=False)
    transfers['intbis'] = pd.to_datetime(transfers['intbis'])

    # Define important variables for stay exclusion.
    relevant_interval = (2006, 2019)
    excl_speciality_at_19b = ['THGCH', 'FA_THGCH']
    start_times_pdms = {'10A OST': dt.datetime(2010, 7, 1, hour=0, minute=0, second=0),
                        '10A WEST': dt.datetime(2011, 2, 1, hour=0, minute=0, second=0),
                        '10B OST': dt.datetime(2010, 7, 1, hour=0, minute=0, second=0),
                        '11A WEST C': dt.datetime(2011, 2, 1, hour=0, minute=0, second=0),
                        '11A WEST SU': dt.datetime(2011, 2, 1, hour=0, minute=0, second=0),
                        '15B OST': dt.datetime(2011, 10, 1, hour=0, minute=0, second=0),
                        '19A OST': dt.datetime(2001, 2, 1, hour=0, minute=0, second=0),
                        '19B OST': dt.datetime(2005, 11, 1, hour=0, minute=0, second=0),
                        'ANAES INT 2': dt.datetime(2003, 5, 1, hour=0, minute=0, second=0),
                        'ANAES PAS': dt.datetime(2001, 2, 1, hour=0, minute=0, second=0),
                        'CHIRURGIE 8': dt.datetime(2012, 4, 1, hour=0, minute=0, second=0),
                        'CHIRURGIE 9': dt.datetime(2099, 1, 1, hour=0, minute=0, second=0)}
    interval_close_transfers = dt.timedelta(hours=12)
    min_observation_period = dt.timedelta(hours=72)

    # Map stations to unique names.
    transfers['station'] = transfers['station'].map(station_mapping)
    assert list(station_mapping.keys()).sort() == stations.sort()
    assert all(transfers[transfers['station'].isin(stations)])
    print_total_transfers(transfers)
    original_transfers = transfers.copy()
    # Ensure that discharge entry is theme for each case.
    assert all(transfers.groupby('fallnummer')['discharge'].nunique() == 1)

    # This routine expects only ICU transfers (these might be included/excluded ICUs).
    # All gaps between these ICU transfers are considered as non-ICU transfers or external/home transfers.
    # Among others, this is necessary to correctly identify subsequent ICU transfers and discharges to excluded ICUs.

    # Add additional information about next ICU for exclusion based on discharge to another ICU.
    # Must be done before removing any transfers to use all transfers for next transfer information.
    print(f"Add information about last transfers for further processing.")
    # Indicates last entry per patient. Should only be used to exclude death.
    transfers[['last_per_patient', 'last_per_case']] = False
    transfers.loc[transfers.groupby(['patientid'])['intbis'].idxmax(), 'last_per_patient'] = True
    assert len(transfers['patientid'].unique().tolist()) == transfers[transfers['last_per_patient']].shape[0]
    transfers.loc[transfers.groupby(['fallnummer'])['intbis'].idxmax(), 'last_per_case'] = True
    assert len(transfers['fallnummer'].unique().tolist()) == transfers[transfers['last_per_case']].shape[0]
    print(f"\tMarked {transfers[transfers['last_per_patient']].shape[0]} transfers as last per patient.")
    print(f"\tMarked {transfers[transfers['last_per_case']].shape[0]} transfers as last per case.")
    print(f"Add information about next transfers for further processing.")
    # Careful: next_trans_* contain information about next transfers of same patient regardless of a new case.
    # Consider all cases of a patient and sort them descending for transfers and the remove info for last transfer.
    transfers.sort_values(['patientid', 'intvon'], ascending=[True, False], inplace=True, ignore_index=True)
    next_columns = ['id', 'fallnummer', 'intvon', 'station', 'speciality']
    next_column_names = ['next_trans_' + column for column in next_columns]
    transfers[next_column_names] = transfers[next_columns].shift(periods=1)
    transfers.loc[transfers['last_per_patient'], next_column_names] = [None for _ in next_column_names]
    assert all((transfers['next_trans_intvon'].isna()) | (transfers['intvon'] < transfers['next_trans_intvon']))
    assert all((transfers['next_trans_intvon'].isna()) | (transfers['intbis'] <= transfers['next_trans_intvon']))

    # Careful: Consecutive and close cases (candidates for merging) only consider the same case of the same patient.
    # Mark consecutive transfers.
    transfers.loc[(~transfers['last_per_case']) & (transfers['fallnummer'] == transfers['next_trans_fallnummer']) &
                  (transfers['intbis'] == transfers['next_trans_intvon']), 'next_trans_consecutive'] = True
    transfers['next_trans_consecutive'] = transfers['next_trans_consecutive'].fillna(False)
    # Mark close transfers.
    transfers.loc[(~transfers['last_per_case']) & (transfers['fallnummer'] == transfers['next_trans_fallnummer']) &
                  (transfers['next_trans_intvon'] - transfers['intbis'] <= interval_close_transfers),
                  'next_trans_close'] = True
    transfers['next_trans_close'] = transfers['next_trans_close'].fillna(False)
    # Previously also marked stays with intervention or surgery in between (value 7/10 for itemid 2201), but turned out
    # to be ineffective since only a single transfer identified in addition to consecutive and close trasnfers.
    print(f"\tMarked {transfers[transfers['next_trans_consecutive']].shape[0]} transfers as consecutive.")
    print(f"\tMarked {transfers[transfers['next_trans_close']].shape[0]} transfers as close.")
    print_total_transfers(transfers)

    # Exclude transfer not in relevant interval.
    old_num_transfers = transfers.shape[0]
    transfers.drop(transfers[transfers['ukmvon'].dt.year < relevant_interval[0]].index, inplace=True)
    print(f"Excluded transfers from before {relevant_interval[0]} (n={old_num_transfers - transfers.shape[0]})")
    old_num_transfers = transfers.shape[0]
    # All transfers not marked included should be excluded by now.
    transfers.drop(transfers[transfers['ukmbis'].dt.year > relevant_interval[1]].index, inplace=True)
    transfers.drop(transfers[transfers['ukmbis'].dt.year.isnull()].index, inplace=True)
    print(f"Excluded transfers until after {relevant_interval[1]} (n={old_num_transfers - transfers.shape[0]})")
    print_total_transfers(transfers)

    # Print statistics of all exported transfers relevant for the study.
    print("Statistics for study:")
    print(f"Number of transfers per station {transfers['station'].value_counts().to_dict()}")
    patient_stations = transfers[['patientid', 'station']].drop_duplicates()
    print(f"Number of patients per station {patient_stations['station'].value_counts().to_dict()}\n")

    # Exclude transfers not managed by the department of anesthesiology.
    old_num_transfers = transfers.shape[0]
    transfers.drop(transfers[~is_in_stations(transfers, anesthesiology_stations)].index, inplace=True)
    print(f"Excluded transfers of stations not managed by anesthesiology (n={old_num_transfers - transfers.shape[0]})")
    old_num_transfers = transfers.shape[0]
    transfers.drop(transfers[~is_not_excl_speciality_on_19b(transfers, excl_speciality_at_19b)].index, inplace=True)
    assert not ((transfers['station'] == '19B OST') &
                (transfers['station'].str.contains('|'.join(excl_speciality_at_19b)))).to_numpy().any()
    print(f"(Exclude transfers at station 19B managed by thoracic surg. (n={old_num_transfers - transfers.shape[0]}))")
    print_total_transfers(transfers)

    # Exclude IMC transfers.
    old_num_transfers = transfers.shape[0]
    transfers.drop(transfers[transfers['station'] == '15B OST'].index, inplace=True)
    print(f"Excluded IM transfers at 15B OST (n={old_num_transfers - transfers.shape[0]})")
    old_num_transfers = transfers.shape[0]
    rm_transfers_per_station = {}
    for station in stations:
        rm_transfers_per_station[station] = \
            transfers[(transfers['station'] == station) & (transfers['intvon'] < start_times_pdms[station])].shape[0]
        transfers.drop(transfers[(transfers['station'] == station) &
                                 (transfers['intvon'] < start_times_pdms[station])].index, inplace=True)
    print(f"For validation: excluded transfers not in PDMS system of department of anesthesiology (QS-system) "
          f"(n={old_num_transfers - transfers.shape[0]}).")
    print("(transfers per station: " + str(rm_transfers_per_station) + ")")
    print_total_transfers(transfers)

    # Check that dates are valid. Only valid after exclusion of observation interval and stations.
    assert not transfers['ukmbis'].isnull().to_numpy().any()
    # Verify inclusion/exclusion criteria.
    assert ((is_in_obs_interval(transfers, relevant_interval) & is_in_stations(transfers, included_stations) &
             is_in_pdms(transfers, start_times_pdms) &
             is_not_excl_speciality_on_19b(transfers, excl_speciality_at_19b))).to_numpy().all()

    # Merge "following" (i.e. consecutive, close)
    # Necessary to loop over data, since arbitrary number of merges can take place.
    old_num_transfers = transfers.shape[0]
    assert transfers[transfers['station'].isin(included_stations)].shape[0] == transfers.shape[0]
    # Consider all transfers that belong to a case in descending order.
    transfers.sort_values(['fallnummer', 'intvon'], ascending=[True, False], inplace=True, ignore_index=True)
    transfers['merged_ids'] = transfers.apply(lambda transfer: [transfer['id']], axis=1)
    case_id = -1
    next_transfer_idx = -1
    # After merging: stations and specialities of included transfers (included icus and not thoracic surgery on 19B)
    # contain enumerations for stations/specialities; other transfers still contain single entries in these fields.
    rm_indices = []
    # Variables to record number of merges.
    num_merged_consecutive = 0
    num_merged_first_very_short = 0
    num_merged_second_very_short = 0
    num_merged_continuous_hf = 0
    num_merged_manually = 0
    num_not_merged_to_normal = 0
    num_not_merged_manually = 0
    for idx, _ in transfers.iterrows():
        if transfers.at[idx, 'fallnummer'] != case_id:
            # New transfer (transfer id) encountered.
            case_id = transfers.at[idx, 'fallnummer']
        else:
            # Same case id encountered. Consider merging.
            if transfers.at[idx, 'next_trans_id'] == transfers.at[next_transfer_idx, 'id']:
                assert transfers.at[next_transfer_idx, 'patientid'] == transfers.at[idx, 'patientid']
                assert transfers.at[next_transfer_idx, 'intvon'] > transfers.at[idx, 'intvon']
                # Verify that really consecutive transfers and not excluded stay removed in between.
                consecutive, close = transfers.loc[idx, ['next_trans_consecutive', 'next_trans_close']]
                num_merged_consecutive += 1 if consecutive else 0
                is_following = False
                if not consecutive and close:
                    # Found two included stays (i.e. both at anaesthesiology icu stations) that are not consecutive with
                    # a gap of interval_close_transfers.
                    case_id = transfers.at[idx, 'fallnummer']
                    icu_start = transfers.at[idx, 'intvon']
                    icu_end = transfers.at[idx, 'intbis']
                    next_icu_start = transfers.at[next_transfer_idx, 'intvon']
                    next_icu_end = transfers.at[next_transfer_idx, 'intbis']

                    max_interval_no_hf = det_max_hf_4h_around_discharge(db_conn, case_id, icu_end, next_icu_start)
                    discharge_target = det_discharge_target_in_pdms(db_conn, case_id, icu_start, icu_end)
                    # 1. First transfer very short, so a too early entry in the system.
                    if icu_end - icu_start <= pd.Timedelta(hours=1):
                        num_merged_first_very_short += 1
                        is_following = True
                    # 2. Next transfer very short, so a too late additional entry in the system.
                    elif next_icu_end - next_icu_start <= pd.Timedelta(hours=1):
                        num_merged_second_very_short += 1
                        is_following = True
                    # 3. Check if at most one hf entry is missing, so that probably no discharge.
                    elif max_interval_no_hf <= pd.Timedelta(minutes=30):
                        num_merged_continuous_hf += 1
                        is_following = True
                    # 4. Determine if clearly not consecutive with >2h interval no hf and discharge in PDMS.
                    elif (max_interval_no_hf > pd.Timedelta(hours=2)) and (discharge_target == 'Allgem.'):
                        num_not_merged_to_normal += 1
                        is_following = False
                    # 5. Manually curated cases.
                    elif case_id in ['26708419', '40696784', '56772790', '57397640', '67622472', '73267463', '75328826',
                                     '75987250']:
                        num_merged_manually += 1
                        is_following = True
                    elif case_id in ['30532848', '39098628', '40730460', '42263087', '42305120', '43738330', '47518288',
                                     '59730622', '61517901', '69811426', '73295653']:
                        num_not_merged_manually += 1
                        is_following = False
                    else:
                        print(transfers.at[idx, 'id'], transfers.at[idx, 'patientid'], transfers.at[idx, 'fallnummer'],
                              max_interval_no_hf, discharge_target)
                        raise Exception(f"No proper handling defined for transfer {transfers.at[idx, 'id']}.")

                # If any of the above criteria is met, merge the corresponding stays.
                if consecutive or (close and is_following):
                    copy_columns = transfers.columns.difference(['id', 'intvon', 'station', 'speciality', 'merged_ids'])
                    transfers.loc[idx, copy_columns] = transfers.loc[next_transfer_idx, copy_columns]
                    merged_ids, station, speciality = transfers.loc[idx, ['merged_ids', 'station', 'speciality']]
                    n_merged_ids, n_station, n_speciality = transfers.loc[next_transfer_idx,
                                                                          ['merged_ids', 'station', 'speciality']]
                    transfers.loc[idx, ['merged_ids', 'station', 'speciality']] =\
                        np.array((merged_ids + n_merged_ids, station + ';' + n_station,
                                  speciality + ';' + n_speciality), dtype=object)
                    rm_indices.append(next_transfer_idx)
        next_transfer_idx = idx
    transfers.drop(rm_indices, inplace=True)
    assert old_num_transfers - transfers.shape[0] == num_merged_consecutive + num_merged_first_very_short + \
           num_merged_second_very_short + num_merged_continuous_hf + num_merged_manually
    print(f"Merged following transfers (n={old_num_transfers - transfers.shape[0]}). "
          f"[consecutive: {num_merged_consecutive}, first <=1h: {num_merged_first_very_short}, second <=1h: "
          f"{num_merged_second_very_short}, cont. hf: {num_merged_continuous_hf}, manual {num_merged_manually}]")
    print(f"[not merged: >2h interval and PDMS discharge {num_not_merged_to_normal}, manual {num_not_merged_manually}]")
    print_total_transfers(transfers)

    # Exclude transfers ending with death.
    old_num_transfers = transfers.shape[0]
    transfers.drop(transfers[(transfers['discharge'] == 'M') & (transfers['intbis'] == transfers['ukmbis']) &
                             (transfers['last_per_patient'])].index, inplace=True)
    print(f"Excluded transfers with death at ICU (n={old_num_transfers - transfers.shape[0]})")
    print_total_transfers(transfers)

    # Exclude transfers at another ICU/IMC not managed by anesthesiology.
    old_num_transfers = transfers.shape[0]
    transfers.drop(transfers[(~next_trans_is_in_pdms(transfers, start_times_pdms)) &
                             transfers['next_trans_consecutive']].index, inplace=True)
    num_excl_consecutive = old_num_transfers - transfers.shape[0]
    # Two stays with the first one included (i.e. at anaesthesiology icu stations) and the next one at an excluded ICU
    # that are not consecutive with a gap of interval_close_transfers.
    num_excl_short_with_target = 0
    num_incl_to_normal = 0
    num_incl_manually = 0
    num_excl_manually = 0
    idx_consecutive = []
    for idx, _ in transfers.loc[(~next_trans_is_in_pdms(transfers, start_times_pdms)) &
                                transfers['next_trans_close']].iterrows():
        case_id = transfers.at[idx, 'fallnummer']
        icu_start = transfers.at[idx, 'intvon']
        icu_end = transfers.at[idx, 'intbis']
        next_icu_start = transfers.at[idx, 'next_trans_intvon']
        next_station = transfers.at[idx, 'next_trans_station']
        discharge_target = det_discharge_target_in_pdms(db_conn, case_id, icu_start, icu_end)
        discharge_mapping = {'10A WEST': '10AW', '11A WEST SU': '10AW', '11A WEST C': '10AW', '15B OST': '15BO',
                             'CHIRURGIE 8': 'CHIR8/9', 'CHIRURGIE 9': 'CHIR8/9'}
        # 1. Interval relatively short (<=2h) and discharge entry in PDMS contains next station.
        if next_icu_start - icu_end <= pd.Timedelta(hours=2) and \
                discharge_target == discharge_mapping.get(next_station, 'empty'):
            num_excl_short_with_target += 1
            idx_consecutive.append(idx)
        # 2. If interval relatively long (>6h) and discharge entry in PDMS to normal station.
        elif next_icu_start - icu_end > pd.Timedelta(hours=6) and discharge_target == 'Allgem.':
            num_incl_to_normal += 1
        # 3. Manually curated cases.
        elif case_id in ['27628036', '30596420', '38466950', '43324977', '47417716', '48721010', '51154240', '57332964',
                         '67110382', '74014569', '80594879']:
            num_excl_manually += 1
            idx_consecutive.append(idx)
        elif case_id in ['27578640', '39741393', '46656202', '49037732', '56407294', '56996940', '58690562', '59182293',
                         '60740259', '61624449', '67204026', '69149944', '70861356', '74539661', '77337369', '77367675',
                         '80373708', '81856699']:
            num_incl_manually += 1
        else:
            print(transfers.at[idx, 'id'], icu_end - icu_start, discharge_target, next_station)
            raise Exception(f"No proper handling defined for ext. transfer {transfers.at[idx, 'id']}.")

    transfers.drop(idx_consecutive, inplace=True)
    assert old_num_transfers - transfers.shape[0] == \
           num_excl_consecutive + num_excl_short_with_target + num_excl_manually
    print(f"Excluded transfers w/ following transfer at ICU/IM man. ext. (n={old_num_transfers - transfers.shape[0]}). "
          f"[consecutive: {num_excl_consecutive}, with short interval and target: {num_excl_short_with_target}, "
          f"manual {num_excl_manually}]")
    print(f"[included: discharge to normal and interval >6h {num_incl_to_normal}, manual {num_incl_manually}]")
    print_total_transfers(transfers)

    # Exclude transfers with insufficient observation period.
    old_num_transfers = transfers.shape[0]
    transfers.drop(transfers[(~((transfers['discharge'] == 'M') & (transfers['last_per_patient']))) &
                             (transfers['intbis'] + min_observation_period > transfers['ukmbis'])].index, inplace=True)
    print(f"Excluded transfers without min observation interval of {min_observation_period} after ICU stay "
          f"(n={old_num_transfers - transfers.shape[0]})")
    deaths_outside_icu_3d = transfers.loc[((transfers['discharge'] == 'M') & (transfers['last_per_patient'])) &
                                          (transfers['intbis'] + min_observation_period > transfers['ukmbis'])].shape[0]
    print(f"(Deaths outside ICU within 3d n={deaths_outside_icu_3d})")
    # All transfers that are observed less that 3d after ICU die at a normal station.
    assert all(transfers.loc[(transfers['last_per_patient']) &
                             (transfers['intbis'] + min_observation_period > transfers['ukmbis']), 'discharge'] == 'M')
    print_total_transfers(transfers)

    # Remove implausible transfers.
    old_num_transfers = transfers.shape[0]
    age_recordings = db_conn.read_records_from_db(['numericvalue'], 96)
    age_recordings['numericvalue'] = pd.to_numeric(age_recordings['numericvalue'])
    age_recordings = age_recordings.loc[age_recordings['numericvalue'] > 0]
    age_recordings.sort_values('displaytime', ascending=True, inplace=True, ignore_index=True)
    transfers = transfers.loc[transfers.apply(
        lambda trans:(age_recordings.loc[(age_recordings['patientid'].isin(
            list(transfers.loc[transfers['patientid'] == trans['patientid'], 'fallnummer']))) &
                                         (age_recordings['displaytime'] <= trans['intbis'])].shape[0] > 0), axis=1)]
    print(f"Excluded transfers without age entry >0 until end of stay (n={old_num_transfers - transfers.shape[0]})")
    age_recordings = db_conn.read_records_from_db(['numericvalue'], 96)
    age_recordings['numericvalue'] = pd.to_numeric(age_recordings['numericvalue'])
    age_recordings = age_recordings.loc[age_recordings['numericvalue'] >= 18]
    age_recordings.sort_values('displaytime', ascending=True, inplace=True, ignore_index=True)
    transfers = transfers.loc[transfers.apply(
        lambda trans:(age_recordings.loc[(age_recordings['patientid'].isin(
            list(transfers.loc[transfers['patientid'] == trans['patientid'], 'fallnummer']))) &
                                         (age_recordings['displaytime'] <= trans['intbis'])].shape[0] > 0), axis=1)]
    print(f"Excluded transfers without age entry >=18 until end of stay (n={old_num_transfers - transfers.shape[0]})")
    old_num_transfers = transfers.shape[0]
    # Based on HF _values_ so only valid measurements especially >0 considered.
    transfers_min_max_hf = db_conn.read_transfers_with_min_max_variable_from_db('Heart rate')
    # Delete all transfers where the transfer itself and all transfers merged into it contain no HF measurement.
    transfers.drop(transfers[transfers['merged_ids'].apply(
        lambda indices: all([j not in transfers_min_max_hf['id'].tolist() for j in indices]))].index, inplace=True)
    print(f"Excluded transfers with no hf during stay (n={old_num_transfers - transfers.shape[0]})")
    # Add min and max hf recording times during all merged transfer.
    transfers['first_hf_rec'] = dt.datetime(2030, 1, 1, hour=0, minute=0, second=0)
    transfers['last_hf_rec'] = dt.datetime(1990, 1, 1, hour=0, minute=0, second=0)
    for idx, _ in transfers.iterrows():
        for i in transfers.loc[idx, 'merged_ids']:
            first_hf_time_during_transfer = transfers_min_max_hf.loc[transfers_min_max_hf['id'] == i, 'min']
            last_hf_time_during_transfer = transfers_min_max_hf.loc[transfers_min_max_hf['id'] == i, 'max']
            if first_hf_time_during_transfer.shape[0] > 0:
                if first_hf_time_during_transfer.iloc[0] < transfers.loc[idx, 'first_hf_rec']:
                    transfers.loc[idx, 'first_hf_rec'] = first_hf_time_during_transfer.iloc[0]
            if last_hf_time_during_transfer.shape[0] > 0:
                if last_hf_time_during_transfer.iloc[0] > transfers.loc[idx, 'last_hf_rec']:
                    transfers.loc[idx, 'last_hf_rec'] = last_hf_time_during_transfer.iloc[0]
    assert all(transfers.loc[:, 'first_hf_rec'] < dt.datetime(2030, 1, 1, hour=0, minute=0, second=0))
    assert all(transfers.loc[:, 'last_hf_rec'] > dt.datetime(1990, 1, 1, hour=0, minute=0, second=0))
    # Remove all transfers with less than 1h of hf recordings.
    old_num_transfers = transfers.shape[0]
    transfers.drop(transfers[transfers['last_hf_rec'] - transfers['first_hf_rec'] < pd.Timedelta(hours=1)].index,
                   inplace=True)
    print(f"Excluded transfers with less than 1h hf rec. during stay (n={old_num_transfers - transfers.shape[0]})")
    # Remove all transfers with less than 2h of hf recordings.
    old_num_transfers = transfers.shape[0]
    transfers.drop(transfers[transfers['last_hf_rec'] - transfers['first_hf_rec'] < pd.Timedelta(hours=2)].index,
                   inplace=True)
    print(f"Excluded transfers with less than 2h hf rec. during stay (n={old_num_transfers - transfers.shape[0]})")
    print_total_transfers(transfers)

    # Correct discharge time to last hf.
    # 1. When interval without hf before discharge <=6h, set discharge to last hf.
    hf_less_6h_till_discharge = (transfers['intbis'] - transfers['last_hf_rec'] > pd.Timedelta(minutes=15)) & \
                                (transfers['intbis'] - transfers['last_hf_rec'] <= pd.Timedelta(hours=6))
    diffs = transfers.loc[hf_less_6h_till_discharge, 'intbis'] - transfers.loc[hf_less_6h_till_discharge, 'last_hf_rec']
    # ids_to_max_diffs = [(transfers.loc[i, 'id'], format_timedelta(td))
    #                     for (td, i) in list(zip(diffs.nlargest(n=200), diffs.nlargest(n=200).index))]
    transfers.loc[hf_less_6h_till_discharge, 'intbis'] = transfers.loc[hf_less_6h_till_discharge, 'last_hf_rec']
    print(f"Change {diffs.shape[0]} discharges <=6h no hf at end to last hf (mean: {diffs.mean()} max: {diffs.max()})")
    # 2. When >6h, check if PDMS discharge coincides with hf and then use the last hf close to PDMS discharge.
    hf_more_6h_till_discharge = (transfers['intbis'] - transfers['last_hf_rec'] > pd.Timedelta(minutes=15)) & \
                                (transfers['intbis'] - transfers['last_hf_rec'] > pd.Timedelta(hours=6))
    transfers['corrected'] = False
    for idx, _ in transfers.loc[hf_more_6h_till_discharge].iterrows():
        case_id = transfers.at[idx, 'fallnummer']
        icu_start = transfers.at[idx, 'intvon']
        icu_end = transfers.at[idx, 'intbis']
        discharge_time = det_discharge_time_in_pdms(db_conn, case_id, icu_start, icu_end)
        if abs(discharge_time - transfers.at[idx, 'last_hf_rec']) <= pd.Timedelta(hours=6):
            transfers.at[idx, 'corrected'] = True
    diffs = transfers.loc[transfers['corrected'], 'intbis'] - transfers.loc[transfers['corrected'], 'last_hf_rec']
    transfers.loc[transfers['corrected'], 'intbis'] = transfers.loc[transfers['corrected'], 'last_hf_rec']
    print(f"Change {diffs.shape[0]} discharges >6h no hf at end to last hf (mean: {diffs.mean()} max: {diffs.max()})")
    old_num_transfers = transfers.shape[0]
    transfers.drop(transfers[hf_more_6h_till_discharge & (~(transfers['corrected']))].index, inplace=True)
    print(f"Removed transfers >6h no hf at end and no discharge in PDMS (n={old_num_transfers - transfers.shape[0]})")
    transfers.drop(columns=['corrected', 'first_hf_rec', 'last_hf_rec'], inplace=True)
    print_total_transfers(transfers)
    # Not necessary anymore.
    # print(f"Try to assign all HF values to stays.")
    # Parallel(n_jobs=4)(delayed(assign_hf_values_to_stays)(args.password, transfers.loc[transfers['patientid'] == pid])
    #                     for pid in transfers['patientid'].unique().tolist())
    # Serial processing for debug purpose.
    # [assign_hf_values_to_stays(args.password, transfers.loc[transfers['patientid'] == pid])
    # for pid in transfers['patientid'].unique().tolist()]

    # Print statistics of all included transfers.
    dict_transfers_station = {'19A OST': 0, '19B OST': 0, 'ANAES INT 2': 0, 'ANAES PAS': 0}
    dict_patients_station = {'19A OST': set(), '19B OST': set(), 'ANAES INT 2': set(), 'ANAES PAS': set()}
    for idx, _ in transfers.iterrows():
        stations_split = transfers.at[idx, 'station'].split(';')
        for station in stations_split:
            dict_transfers_station[station] += 1
            dict_patients_station[station].add(transfers.at[idx, 'patientid'])
    for station, patienids in dict_patients_station.items():
        dict_patients_station[station] = len(patienids)
    print(f"(Number of transfers per station {dict_transfers_station})")
    print(f"(Number of patients per station {dict_patients_station})")
    print("")

    db_conn.write_stays_to_db(transfers)

    # Debug code for unprocessed stays
    # case_number = 0
    # for idx, _ in unprocessed_transfers.iterrows():
    #     case_number += 1
    #     case_id = unprocessed_transfers.at[idx, 'fallnummer']
    #     icu_start = unprocessed_transfers.at[idx, 'intvon']
    #     icu_end = unprocessed_transfers.at[idx, 'intbis']
    #     discharge_target = det_discharge_target_in_pdms(db_conn, case_id, icu_start, icu_end)
    #     print(f"case {case_number};Fallnummer: {unprocessed_transfers.at[idx, 'fallnummer']}"
    #           f"(patientid: {unprocessed_transfers.at[idx, 'patientid']})\n"
    #           f";Entlassung {unprocessed_transfers.at[idx, 'station']}: "
    #           f"{unprocessed_transfers.at[idx, 'intbis'].strftime('%d.%m.%Y %H:%M:%S')}\n"
    #           f";Aufnahme {unprocessed_transfers.at[idx, 'next_trans_station']}: "
    #           f"{unprocessed_transfers.at[idx, 'next_trans_intvon'].strftime('%d.%m.%Y %H:%M:%S')}\n"
    #           f";Intervall: "
    #           f"{unprocessed_transfers.at[idx, 'next_trans_intvon'] - unprocessed_transfers.at[idx, 'intbis']}\n"
    #           f";QS discharge: {discharge_target}\n;;;")
    #     patient_transfers = \
    #         original_transfers.loc[original_transfers['patientid'] == unprocessed_transfers.at[idx,
    #                                                                                            'patientid']].copy()
    #     patient_transfers.sort_values(['fallnummer', 'intvon'], ascending=[True, True], inplace=True,
    #                                   ignore_index=True)
    #     patient_transfers['mark'] = ' '
    #     patient_transfers.loc[patient_transfers['id'] == unprocessed_transfers.at[idx, 'next_trans_id'],
    #                           'mark'] = '<--'
    #     ids = json.loads(str(unprocessed_transfers.at[idx, 'merged_ids']))
    #     patient_transfers.loc[patient_transfers['id'] == max(ids), 'mark'] = '<--'
    #     patient_transfers.drop(columns=['id'], inplace=True)
    #     patient_transfers.to_csv(sys.stdout, sep=';')
    #     print(';;;')
    #     print(';;;')


@output_wrapper
def assign_hf_values_to_stays(password, patient_transfers):
    # Stays are expected to have a hf-free interval at start and end since they were already merged.
    # This routine detects these breaks as new start and ends of a stay.
    # First usage showed that discharges are actually quite close to hf stop and that it is not necessary.
    db_conn = ResearchDBConnection(password)
    assert patient_transfers['patientid'].nunique() == 1
    pid = patient_transfers.iloc[0]['patientid']
    print(pid)
    case_ids = patient_transfers['fallnummer'].unique().tolist()
    hf_values = db_conn.read_values_from_db('Heart rate', 'continuous', case_id=case_ids[0])
    for i in range(1, len(case_ids)):
        hf_values = hf_values.append(db_conn.read_values_from_db('Heart rate', 'continuous', case_id=case_ids[i]),
                                     ignore_index=True)
    hf_values.sort_values(['displaytime'], ascending=[True], inplace=True, ignore_index=True)
    patient_transfers.sort_values(['intvon'], ascending=[True], inplace=True, ignore_index=True)
    hf_values['delta_start'] = hf_values['displaytime'] - hf_values['displaytime'].shift(1)
    hf_values['delta_end'] = abs(hf_values['displaytime'] - hf_values['displaytime'].shift(-1))
    hf_values[['dist_start', 'dist_end']] = 0
    for idx, _ in patient_transfers.iterrows():
        hf_values['dist_start'] = abs(hf_values['displaytime'] - patient_transfers.loc[idx, 'intvon'])
        hf_values['dist_end'] = abs(hf_values['displaytime'] - patient_transfers.loc[idx, 'intbis'])
        closest_30min_gap_to_start =\
            hf_values.loc[hf_values.loc[(hf_values['delta_start'] > pd.Timedelta(minutes=30)) |
                                        (hf_values['delta_start'].isnull()), 'dist_start'].idxmin(), 'displaytime']
        closest_30min_gap_to_end =\
            hf_values.loc[hf_values.loc[(hf_values['delta_end'] > pd.Timedelta(minutes=30)) |
                                        (hf_values['delta_end'].isnull()), 'dist_end'].idxmin(), 'displaytime']
        assert closest_30min_gap_to_start < closest_30min_gap_to_end
        patient_transfers.loc[idx, 'first_hf'] = closest_30min_gap_to_start
        patient_transfers.loc[idx, 'last_hf'] = closest_30min_gap_to_end

    prev_end = None
    for idx, _ in patient_transfers.iterrows():
        # Control that all transfers still not overlapping.
        assert prev_end is None or (prev_end < patient_transfers.loc[idx, 'first_hf'])
        prev_end = patient_transfers.loc[idx, 'last_hf']
        if type(patient_transfers.loc[idx, 'first_hf']) is not np.float64:
            distance_from_start = patient_transfers.loc[idx, 'first_hf'] - patient_transfers.loc[idx, 'intvon']
            if abs(distance_from_start) > dt.timedelta(hours=4):
                print(f"First HF {distance_from_start} off from ICU admission")
        if type(patient_transfers.loc[idx, 'last_hf']) is not np.float64:
            distance_from_end = patient_transfers.loc[idx, 'last_hf'] - patient_transfers.loc[idx, 'intbis']
            if abs(distance_from_end) > dt.timedelta(hours=4):
                print(f"Last HF {distance_from_end} off from ICU discharge")

    return patient_transfers


if __name__ == '__main__':
    main()
