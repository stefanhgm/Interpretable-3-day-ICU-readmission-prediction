"""
Script to perform global preprocessing steps on UKM data obtained from the PDMS.
"""
import argparse
import datetime as dt

from psycopg2 import sql
import pandas as pd

from research_database.research_database_communication import ResearchDBConnection


def print_records_changes(cur, db_conn, old_num_records, old_num_records_per_item):
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    cur.execute(sql.SQL("select count(*) from public.recordings;"))
    print(f"Removed {old_num_records - cur.fetchone()[0]} records.")
    counts = old_num_records_per_item.join(
        db_conn.read_records_per_item().set_index('itemid'), on='itemid', rsuffix='_new')
    counts['count_new'] = counts['count_new'].fillna(0)
    counts['removed'] = counts['count'] - counts['count_new']
    print(f"Summed removals {counts['removed'].sum()}")
    counts.drop(['itemid', 'itemtype_new', 'name_new', 'count_new'], axis=1, inplace=True)
    counts.sort_values('removed', ascending=False, inplace=True, ignore_index=True)
    print(counts.loc[counts['removed'] != 0])


def main():
    parser = argparse.ArgumentParser(description='Perform global pre-processing steps.')
    parser.add_argument('password', type=str, help='Database password.')
    args, _ = parser.parse_known_args()

    db_conn = ResearchDBConnection(args.password)
    cur, conn = db_conn.get_cur_conn()

    print("1a. Remove duplicated records for non-medication entries with same values. Duplicates with different values"
          " dealt with during item processing.")
    cur.execute(sql.SQL("select count(*) from public.recordings;"))
    old_num_records = cur.fetchone()[0]
    old_num_records_per_item = db_conn.read_records_per_item()
    # From https://stackoverflow.com/questions/26769454/how-to-delete-duplicate-rows-without-unique-identifier/46775289
    cur.execute(sql.SQL("delete from recordings R1 using recordings R2 " +
                        "where R1.id < R2.id "
                        "and R1.patientid         = R2.patientid " +
                        "and R1.itemid            = R2.itemid " +
                        "and R1.displaytime       = R2.displaytime " +
                        "and R1.numericvalue_text = R2.numericvalue_text " +
                        "and R1.numericvalue      = R2.numericvalue " +
                        "and R1.textvalue_text    = R2.textvalue_text " +
                        "and R1.textvalue         = R2.textvalue "
                        "and R1.itemid not in (select id from items where itemtype=10 or itemtype=11 or itemtype=12);"))
    conn.commit()
    print_records_changes(cur, db_conn, old_num_records, old_num_records_per_item)

    print("1b. Remove duplicate records for non-medications with different text values.")
    cur.execute(sql.SQL("select count(*) from public.recordings;"))
    old_num_records = cur.fetchone()[0]
    old_num_records_per_item = db_conn.read_records_per_item()
    cur.execute(sql.SQL("delete from recordings R1 using recordings R2 " +
                        "where R1.id < R2.id "
                        "and R1.patientid       = R2.patientid " +
                        "and R1.itemid            = R2.itemid " +
                        "and R1.displaytime       = R2.displaytime " +
                        "and R1.itemid not in (select id from items where itemtype=10 or itemtype=11 or itemtype=12 or "
                        "itemtype=4 or itemtype=6);"))
    conn.commit()
    print_records_changes(cur, db_conn, old_num_records, old_num_records_per_item)

    print("1c. Remove duplicate records for non-medications with different numerical values and >5% SD difference.")
    cur.execute(sql.SQL("select count(*) from public.recordings;"))
    old_num_records = cur.fetchone()[0]
    old_num_records_per_item = db_conn.read_records_per_item()
    # Subquery duplicates: Return all duplicates of item type 4 and 6 grouped by patientid, itemid, displaytime with
    # their mean and sd joined with them item-wide mean and sd.
    # Then delete all records that are part of a duplicate and after that add mean of duplicates w/ sd <5% item-wide sd.
    cur.execute(sql.SQL(
        "with duplicates as ( "
        "select * from "
        "(select patientid, itemid, displaytime, avg(numericvalue::numeric) as mean, "
        "stddev(numericvalue::numeric) as sd, count(*) "
        "from recordings "
        "where itemid in (select id from public.items where itemtype=4 or itemtype=6) "
        "group by patientid, itemid, displaytime "
        "having count(*) > 1) as r1 "
        "left join  "
        "(select itemid as item_itemid, avg(numericvalue::numeric) as item_mean, "
        "stddev(numericvalue::numeric) as item_sd "
        "from recordings "
        "where itemid in (select id from public.items where itemtype=4 or itemtype=6) "
        "group by itemid) as r2 "
        "on r1.itemid = r2.item_itemid "
        "), "
        "delete_duplicates as ( "
        "delete from recordings "
        "where (patientid, itemid, displaytime) in (select patientid, itemid, displaytime from duplicates) "
        ") "
        "insert into "
        "recordings(patientid, itemid, displaytime, numericvalue_text, numericvalue, textvalue_text, textvalue) ( "
        "select patientid, itemid, displaytime, "
        "replace(trim(trailing '.' from trim(trailing '0' from mean::text)), '.', ',') as numericvalue_text, "
        "trim(trailing '.' from trim(trailing '0' from mean::text)) as numericvalue, "
        "\'\\N\' as textvalue_text, \'\\N\' as textvalue "
        "from duplicates d "
        "where d.sd < 0.05 * d.item_sd) returning itemid;"))
    conn.commit()
    inserted_item_ids = cur.fetchall()
    inserted_item_ids_counts = dict()
    for item_id in inserted_item_ids:
        inserted_item_ids_counts[item_id[0]] = inserted_item_ids_counts.get(item_id[0], 0) + 1
    old_num_records_per_item['inserted'] = 0
    for key, value in inserted_item_ids_counts.items():
        old_num_records_per_item.loc[old_num_records_per_item['itemid'] == key, 'inserted'] = value
    print_records_changes(cur, db_conn, old_num_records, old_num_records_per_item)
    print("Duplicates after merging are dealt with during variable processing")

    print("2. Repair, if possible, or remove erroneous hospital admissions (ukmvon, ukmbis).")
    db_conn = ResearchDBConnection(args.password)
    transfers = db_conn.read_table_from_db('transfers')
    original_num_transfers = transfers.shape[0]
    original_num_patients = len(set(transfers['patientid'].to_list()))

    print("2a. Remove hospital admission with discharge after admission.")
    bad_patients = set(transfers.loc[transfers['ukmvon'] >= transfers['ukmbis'], 'patientid'].to_list())
    old_num_transfers = transfers.shape[0]
    transfers.drop(transfers[transfers['patientid'].isin(bad_patients)].index, inplace=True)
    print(f"\tRemoved {len(bad_patients)} patients with {old_num_transfers - transfers.shape[0]} transfers.")

    print("2b. Repair hospital discharge date 2099-12-31 00:00:00 and location when correct data exists for the case.")
    transfers.sort_values(['fallnummer', 'ukmvon', 'ukmbis'],  ascending=[True, True, True], inplace=True,
                          ignore_index=True)
    num_repaired_transfers = 0
    prev_transfer_idx = -1
    for idx, _ in transfers.iterrows():
        if prev_transfer_idx != -1 and \
                transfers.at[idx, 'fallnummer'] == transfers.at[prev_transfer_idx, 'fallnummer'] and \
                transfers.at[idx, 'ukmvon'] == transfers.at[prev_transfer_idx, 'ukmvon'] and \
                transfers.at[prev_transfer_idx, 'ukmbis'] < dt.datetime(2099, 12, 31) and \
                transfers.at[idx, 'ukmbis'] == dt.datetime(2099, 12, 31):
            transfers.loc[idx, 'ukmbis'] = transfers.at[prev_transfer_idx, 'ukmbis']
            transfers.loc[idx, 'discharge'] = transfers.at[prev_transfer_idx, 'discharge']
            num_repaired_transfers += 1
        prev_transfer_idx = idx
    print(f"\tRepaired {num_repaired_transfers} transfer entries.")

    print("2c. Remove remaining transfers with hospital discharge 2099-12-31 00:00:00, since no correct discharge "
          "information can be obtained.")
    bad_cases = transfers.loc[transfers['ukmbis'] == dt.datetime(2099, 12, 31), 'fallnummer'].to_list()
    old_num_transfers = transfers.shape[0]
    transfers.drop(transfers[transfers['fallnummer'].isin(bad_cases)].index, inplace=True)
    print(f"\tRemoved {old_num_transfers - transfers.shape[0]} transfers with bad discharge date.")
    assert transfers.loc[transfers['ukmbis'] == dt.datetime(2099, 12, 31)].shape[0] == 0

    print("2d. Verify that all cases have the same hospital admission and discharge dates and targets.")
    admissions = transfers[['patientid', 'fallnummer', 'ukmvon', 'ukmbis', 'discharge']].drop_duplicates()
    assert admissions.shape[0] == len(set(transfers['fallnummer'].to_list()))

    print("2e. Remove patients with overlapping hospital admissions. Unambiguous repair impossible.")
    admissions.sort_values(['patientid', 'ukmvon', 'ukmbis'], ascending=[True, False, False], inplace=True,
                           ignore_index=True)
    bad_patients = set()
    next_admission_idx = -1
    for idx, _ in admissions.iterrows():
        if next_admission_idx != -1 and \
                admissions.at[idx, 'patientid'] == admissions.at[next_admission_idx, 'patientid'] and \
                admissions.at[idx, 'ukmvon'] < admissions.at[next_admission_idx, 'ukmvon'] \
                < admissions.at[idx, 'ukmbis']:
            bad_patients.add(admissions.at[idx, 'patientid'])
        next_admission_idx = idx
    old_num_transfers = transfers.shape[0]
    transfers.drop(transfers[transfers['patientid'].isin(bad_patients)].index, inplace=True)
    print(f"\tRemoved {len(bad_patients)} patients with {old_num_transfers - transfers.shape[0]} transfers.")

    print("2f. Remove patient that was not distinguishable during label review.")
    old_num_transfers = transfers.shape[0]
    transfers.drop(transfers[transfers['patientid'] == '60054292'].index, inplace=True)
    print(f"\tRemoved patient with {old_num_transfers - transfers.shape[0]} transfers.")

    print("3. Repair, if possible, or remove erroneous hospital transfers (intvon, intbis).")

    print("3a. Remove transfers with admission after discharge.")
    bad_patients = set(transfers.loc[transfers['intvon'] >= transfers['intbis'], 'patientid'].to_list())
    old_num_transfers = transfers.shape[0]
    transfers.drop(transfers[transfers['patientid'].isin(bad_patients)].index, inplace=True)
    print(f"\tRemoved {len(bad_patients)} patients with {old_num_transfers - transfers.shape[0]} transfers.")

    print("3b. Verify that all transfers within their hospital admission.")
    outside_admission =\
        transfers.loc[(transfers['intvon'] > transfers['intbis']) &
                      ((transfers['intvon'] >= transfers['ukmvon']) & (transfers['intvon'] < transfers['ukmbis'])) &
                      ((transfers['intbis'] > transfers['ukmvon']) & (transfers['intbis'] <= transfers['ukmbis']))]
    assert outside_admission.shape[0] == 0

    print("3c. Remove patients with overlapping transfers. Repair might be possible, but only few so safer to remove.")
    transfers.sort_values(['patientid', 'intvon', 'intbis'], ascending=[True, False, False], inplace=True,
                          ignore_index=True)
    while True:
        # Repeat procedure until no problematic transfer detected. There might be nested problems.
        # Deal with all cases of overlapping transfers.
        patient_id = -1
        next_transfer_idx = -1
        bad_patients = set()
        num_completely = 0
        num_same_start = 0
        num_diff_start = 0
        for idx, _ in transfers.iterrows():
            if transfers.at[idx, 'patientid'] != patient_id:
                # New patient encountered.
                patient_id = transfers.at[idx, 'patientid']
            else:
                # Reset patient id to ensure that surrounding of this entry non edited twice.
                patient_id = -1
                # Case 1: Earlier stay completely contains following one
                # Next stay:   |    |
                # Prev stay: |        |
                if transfers.at[idx, 'intvon'] < transfers.at[next_transfer_idx, 'intvon'] and \
                        transfers.at[idx, 'intbis'] > transfers.at[next_transfer_idx, 'intbis']:
                    num_completely += 1
                    bad_patients.add(transfers.at[idx, 'patientid'])
                # Case 2: Same start
                # Next stay:  |     |    /  |     |
                # Prev stay:  |       |  /  |     |
                elif transfers.at[idx, 'intvon'] == transfers.at[next_transfer_idx, 'intvon']:
                    same_end = transfers.at[idx, 'intbis'] == transfers.at[next_transfer_idx, 'intbis']
                    num_same_start += 1
                    bad_patients.add(transfers.at[idx, 'patientid'])
                # Case 3: Different start
                # Next stay:    |     |  /    |   |
                # Prev stay:  |     |    /  |     |
                elif transfers.at[idx, 'intvon'] < transfers.at[next_transfer_idx, 'intvon'] \
                        < transfers.at[idx, 'intbis']:
                    same_end = transfers.at[idx, 'intbis'] == transfers.at[next_transfer_idx, 'intbis']
                    num_diff_start += 1
                    bad_patients.add(transfers.at[idx, 'patientid'])
                    # Earlier: Picked longer entry.
            next_transfer_idx = idx
        if len(bad_patients) == 0:
            break
        old_num_transfers = transfers.shape[0]
        transfers.drop(transfers[transfers['patientid'].isin(bad_patients)].index, inplace=True)
        print(f"\tRemoved {len(bad_patients)} patients with {old_num_transfers - transfers.shape[0]} transfers.")
        print(f"[Types: {num_completely} completely, {num_same_start} same start, {num_diff_start} diff start]")

    transfers.sort_values(['id'], ascending=[True], inplace=True, ignore_index=True)
    num_patients = len(set(transfers['patientid'].to_list()))
    num_transfers = transfers.shape[0]
    print(f"In total removed {original_num_patients - num_patients} of {original_num_patients} patients.")
    print(f"In total removed {original_num_transfers - num_transfers} of {original_num_transfers} transfers.")
    db_conn.write_table_to_db('transfers', transfers)


if __name__ == '__main__':
    main()
