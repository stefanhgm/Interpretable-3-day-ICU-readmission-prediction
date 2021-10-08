"""
Script to convert the raw recordings of items collected in the PDMS into variables and values of these variables.
It contains a general part that is derived automatically from the description of items (e.g. value range) and a
specific part that contains specific preprocessing for certain items.
"""
import argparse
import collections
import json
import re

from joblib import Parallel, delayed
from pandas.core.dtypes.common import is_bool_dtype

from helper.io import read_item_processing_descriptions_from_excel
from helper.util import output_wrapper, parse_time, format_timedelta
from research_database.research_database_communication import ResearchDBConnection

import pandas as pd
import numpy as np
import datetime as dt


def main():
    parser = argparse.ArgumentParser(description='Convert raw item recordings to variables and values.')
    parser.add_argument('password', type=str, help='Database password.')
    parser.add_argument('item_overview', type=str, help='Description of all PDMS items and generated variables.')
    parser.add_argument('--variable_names', nargs='+', help='Only convert specified variables.', default='')
    args, _ = parser.parse_known_args()

    item_processing = read_item_processing_descriptions_from_excel(args.item_overview)

    # Parse "included" items and merged items as specified by variable description.
    indices = item_processing.index[
        ((item_processing['decision'] == 'included') | (item_processing['id'] >= 10000)) &
        (not args.variable_names or item_processing['variable_name'].str.contains('|'.join(args.variable_names)))]
    print(f"Found {len(list(indices))} variables to process.")
    Parallel(n_jobs=40)(delayed(parallel_variable_processing)(args.password, i, idx, item_processing)
                        for i, idx in enumerate(list(indices)))
    # Serial processing for debug purpose.
    # [parallel_variable_processing(args.password, i, idx, item_processing) for i, idx in enumerate(list(indices))]


@output_wrapper
def parallel_variable_processing(password, i, idx, item_processing):
    db_conn = ResearchDBConnection(password)

    # Get all records associated with the item for this variable.
    variable_description = item_processing.loc[idx]
    item_id, var_name, var_type = variable_description.loc[['id', 'variable_name', 'type']]

    if item_id < 10000:
        # Original item stored in the PDMS.
        print(f"Process variable ({i+1}): {var_name} (type: {var_type}, variable_id: {item_id}).")
        var_source_columns = json.loads(variable_description['source_columns'])
        records = db_conn.read_records_from_db(var_source_columns, item_id=item_id)
        records = process_pre_procedures(variable_description['general_pre_procedures'], records)
    elif item_id < 20000:
        # Merged item with self assigned id.
        print(f"Process merged variable ({i+1}): {var_name} (type: {var_type}, variable_id: {item_id}).")
        # Read item recordings based on source_columns.
        # Identify all items that get merged into this variable and read in their source_columns.
        merging_variables = item_processing.loc[(item_processing['decision'] == 'merged') &
                                                (item_processing['variable_name'] == var_name)]
        records = load_merging_variables(db_conn, variable_description, merging_variables)
        # Merging is responsible to reduce multiple columns with data to a single one at position four.
        records = merge_item_to_variable(var_name, item_id, records)

    elif item_id < 30000:
        # Pharmaceutical.
        # print(f"Processed pharmaceutical: {var_name} (variable_id: {item_id}).")
        print(f"Process pharma-category variable ({i+1}): {var_name.split(' ', 1)[1]}.")
        # Use first word of variable name as ATC prefix and included all non-excluded with this prefix.
        atc_prefix = var_name.split()[0]
        merging_variables =\
            item_processing.loc[(item_processing['ATC'].str.contains('^(?:' + atc_prefix + ')', na=False)) &
                                (~item_processing['decision'].str.contains('excluded', na=False))]
        records = load_merging_variables(db_conn, variable_description, merging_variables)
    else:
        raise Exception(f"Invalid variable id{item_id}.")

    records = records.rename(columns={records.columns[3]: 'data'})
    assert len(records.columns) == 4
    records = process_values_and_datatype(var_type, variable_description['values'], records)
    records = process_post_procedures(variable_description['general_post_procedures'], records)

    # For merged variables excluding pharmaceuticals and fluids remove duplicates after merging.
    if 10000 <= item_id < 14000 or 15000 <= item_id < 20000:
        # Remove possible new duplicates: when numerical choose median o.w. remove different values.
        old_num_records = records.shape[0]
        if var_type == 'categorical':
            records.drop_duplicates(inplace=True, ignore_index=True, keep='first')
            temp_num_records = records.shape[0]
            records.drop_duplicates(subset=['patientid', 'itemid', 'displaytime'], keep=False, inplace=True,
                                    ignore_index=True)
            print(f"\tRemoved {old_num_records - temp_num_records} with same and "
                  f"{temp_num_records - records.shape[0]} with different {var_type} duplicates after merging.")
        elif var_type == 'continuous':
            records[records.columns[3]] = records[records.columns[3]].astype(np.float32)
            records = records.groupby(['patientid', 'itemid', 'displaytime']).median().reset_index()
            print(f"\tRemoved {old_num_records - records.shape[0]} {var_type} duplicates after merging.")

    db_conn.write_variable_and_values_to_db(var_name, var_type, records)


def load_merging_variables(db_conn, variable_description, merging_variables):
    merged_item_records = []
    for _, merged_item in merging_variables.iterrows():
        merged_item_source_columns = json.loads(merged_item['source_columns'])
        records = db_conn.read_records_from_db(merged_item_source_columns, item_id=merged_item['id'])
        records = process_pre_procedures(merged_item['general_pre_procedures'], records)
        # Special treatment of pharma variables detected by ATC code.
        if 20000 <= variable_description['id'] <= 30000 and merged_item['QS_ITEMTYPE'] == 10:
            records = process_pharma_variable(merged_item, records)
        merged_item_records.append(records)
    records = pd.concat(merged_item_records, axis=0, ignore_index=True)
    del merged_item_records
    return records


def process_pharma_variable(variable_description, records):
    var_id = variable_description['id']
    var_name = variable_description['QS_NAME']
    atc_code = variable_description['ATC']
    atc_name = variable_description['ATC-name']
    atc_path = variable_description['ATC-path']
    data_column = records.columns.tolist()[3]
    records[data_column] = pd.to_numeric(records[data_column])

    # 1. Remove zero and negative entries.
    old_num_records = records.shape[0]
    records.drop(records[records[data_column] == 0].index, inplace=True)
    num_zeros = old_num_records - records.shape[0]
    old_num_records = records.shape[0]
    records.drop(records[records[data_column] < 0].index, inplace=True)
    num_negative = old_num_records - records.shape[0]
    if records.shape[0] == 0:
        return records

    # 2. Determine median daily dose in the data to normalize the entries and median administration interval.
    records['displaytime_day'] = pd.to_datetime(records['displaytime']).dt.date
    median_ddd = records[['patientid', 'displaytime_day', data_column]].groupby(['patientid', 'displaytime_day']).agg(
        {data_column: 'sum'})[data_column].median()
    records.sort_values(['patientid', 'displaytime'], ascending=[True, True], inplace=True, ignore_index=True)
    records['intervals'] = records.groupby(['patientid'])['displaytime'].shift(-1)
    records['intervals'] = (records['intervals'] - records['displaytime'])
    if not all(records['intervals'].isnull()):
        median_administration_interval = records['intervals'].median(skipna=True)
    else:
        median_administration_interval = pd.Timedelta(0, unit='seconds')

    # 3. Determine whiskers (1.5 of IQR) as too extreme values and characterize them.
    median = records[data_column].median()
    q_025 = records[data_column].quantile(0.25)
    q_075 = records[data_column].quantile(0.75)
    lower_bound = q_025 / 10
    upper_bound = q_075 * 10
    lower_bounds = records.loc[records[data_column] < lower_bound]
    upper_bounds = records.loc[records[data_column] > upper_bound]
    percentage_lower_bounds = lower_bounds.shape[0] * 100 / records.shape[0]
    percentage_upper_bounds = upper_bounds.shape[0] * 100 / records.shape[0]
    median_lower_bounds = lower_bounds[data_column].median()
    median_upper_bounds = upper_bounds[data_column].median()

    # 4. Determine comments and print details of pharmaceutical processing for manual review.
    review_comments = []
    if percentage_lower_bounds > 1:
        review_comments.append("More than 1% of values < 0.1 x 25%-quartile")
    if percentage_upper_bounds > 1:
        review_comments.append("More than 1% of values > 10 x 75%-quartile")
    print(f"\t{var_id}; {var_name}; {atc_code}; {atc_name}; "  # {atc_path}
          f"n={records.shape[0]} (removed zero: {num_zeros}, neg: {num_negative}); "
          f"{format_timedelta(median_administration_interval.round('T'))}; {median_ddd:.3f}/day; "
          f"{percentage_lower_bounds:.1f}%", end='')
    print(f" ({median_lower_bounds:.3f}); " if not pd.isna(median_lower_bounds) else f"; ", end='')
    print(f"{q_025:.3f}; {median:.3f}; {q_075:.3f}; "
          f"{percentage_upper_bounds:.1f}%", end='')
    print(f" ({median_upper_bounds:.3f}); " if not pd.isna(median_upper_bounds) else f"; ", end='')
    print(f"{', '.join(review_comments)}")
    # Changes on dosages in general not used anymore.
    # Change whiskers to median +/- 1.5 IQR.
    # records.loc[records[data_column] < lower_bound, data_column] = lower_bound
    # records.loc[records[data_column] > upper_bound, data_column] = upper_bound
    # assert median == records[data_column].median()
    # assert q_025 == records[data_column].quantile(0.25)
    # assert q_075 == records[data_column].quantile(0.75)
    # print(f"\tChanged {lower_bounds.shape[0] + upper_bounds.shape[0]} extreme to 0.1/10 of 25%/75%-quartiles and "
    #       f"removed {old_num_records - records.shape[0]} zero entries.")
    # Change values into ratios of median ddd to normalize them.
    # records[data_column] = records[data_column] / median_ddd

    # 5. Use item ids as indicators that pharmaceutical was given.
    records[data_column] = f"{var_name} ({atc_code})"

    records.drop(columns=['displaytime_day', 'intervals'], inplace=True)
    records.reset_index(inplace=True, drop=True)

    return records


def process_values_and_datatype(var_type, var_values, records):

    # 1. Process values columns. Note that continuous mapping with dictionary is the same as for categorical variables.
    if var_type == 'categorical' or (var_type == 'continuous' and var_values and var_values.startswith('{')):
        if not is_bool_dtype(records['data']):
            # Remove leading and trailing whitespaces of string values.
            try:
                records['data'] = records['data'].str.strip()
            except AttributeError:
                # In case no all values are strings.
                pass

        var_allowed_values = []
        if var_values.startswith('{'):
            # Categorical values defined by an ordered dictionary or continuous values with mapping.
            # From: https://stackoverflow.com/questions/16641110/converting-string-to-ordered-dictionary
            mapping = json.loads(var_values, object_pairs_hook=collections.OrderedDict)
            # Replace unknown/Unknown in mapping with None.
            while 'unknown' in mapping.values():
                mapping[list(mapping.keys())[list(mapping.values()).index('unknown')]] = None
            while 'Unknown' in mapping.values():
                mapping[list(mapping.keys())[list(mapping.values()).index('Unknown')]] = None
            records['data'] = records['data'].map(lambda x: mapping.get(x, x))
            var_allowed_values = list(mapping.values())
        elif var_values.startswith('['):
            # Categorical values defined by list.
            var_allowed_values = json.loads(var_values)
            # Replace unknown/Unknown in mapping with None.
            var_allowed_values =\
                [value if (value != 'unknown' and value != 'Unknown') else None for value in var_allowed_values]
        elif var_values:
            raise Exception("Unexpected formatting of categorical values.")

        # Only keep records with defined values.
        if len(var_allowed_values) == 0:
            print(f"\tNot range/set of allowed values provided, so assume all are valid.")
        else:
            old_num_records = records.shape[0]
            unique_counts = records.loc[~records['data'].isin(var_allowed_values), 'data'].value_counts(dropna=False)
            records.drop(records[~records['data'].isin(var_allowed_values)].index, inplace=True)
            if unique_counts.shape[0] > 10:
                print(f"\tRemoved {old_num_records - records.shape[0]} records with undefined values "
                      f"({unique_counts.shape[0]} values).")
            else:
                print(f"\tRemoved {old_num_records - records.shape[0]} records with undefined values: "
                      f"{unique_counts.to_dict()}")

    elif var_type == 'continuous' and var_values and var_values.startswith('['):
        # Only keep records in defined value range.
        old_num_records = records.shape[0]
        records['data'] = records['data'].astype(np.float32)
        var_range = json.loads(var_values)
        records.drop(records[records['data'] < var_range[0]].index, inplace=True)
        temp_records = records.shape[0]
        records.drop(records[records['data'] > var_range[1]].index, inplace=True)
        print(f"\tRemoved {old_num_records - temp_records} lower, {temp_records - records.shape[0]} higher records "
              f" outside interval [{var_range[0]}, {var_range[1]}]")

    elif var_values:
        raise Exception("Unexpected formatting of values entry.")

    # 2. Process the datatype.
    if var_type == 'categorical':
        # Use text field in values to represent categorical values.
        records = records.rename(columns={'data': 'textvalue'})
    elif var_type == 'continuous':
        # Use numeric value field to represent continuous values and transform possibly mapped values.
        records['data'] = records['data'].astype(np.float32)
        records = records.rename(columns={'data': 'numericvalue'})
    else:
        raise Exception(f"Unexpected variable type {var_type}.")

    return records


def process_pre_procedures(var_procedures, records):

    procedures = [] if not var_procedures else json.loads(var_procedures)

    for _, procedure in enumerate(procedures):
        # Apply all specified procedures after another.

        if procedure == 'last':
            records = records.sort_values(by=['patientid', 'displaytime'], ascending=True)
            old_num_records = records.shape[0]
            records = records.drop_duplicates(subset=['patientid'], keep='last')
            print(f"\tRemoved {old_num_records - records.shape[0]} records when keeping only last recorded values.")

        elif procedure == 'lower':
            # Set all columns other than itemid, patientid, displaytime that contain strings to lowercase.
            for column in records.columns.tolist():
                if column != 'itemid' and column != 'patientid' and column != 'displaytime':
                    if records[column].dtype == object:
                        records[column] = records[column].str.lower()

        elif procedure.startswith('subtract_'):
            term = float(procedure.split('_')[1])
            data_column = records.columns.tolist()[3]
            records[data_column] = pd.to_numeric(records[data_column]) - term

        elif procedure.startswith('multiply_'):
            factor = float(procedure.split('_')[1])
            data_column = records.columns.tolist()[3]
            records[data_column] = pd.to_numeric(records[data_column]) * factor

        elif procedure.startswith('integrate_'):
            # Integrate function for given time intervals prospectively.
            old_num_records = records.shape[0]
            interval = parse_time(procedure.split('_')[1])
            data_column = records.columns.tolist()[3]
            records.sort_values(['patientid', 'displaytime'], ascending=[True, True], inplace=True, ignore_index=True)
            records['intervals'] = records.groupby(['patientid'])['displaytime'].shift(-1)
            records['intervals'] = (records['intervals'] - records['displaytime'])/interval
            # Only incorporate time spans shorter or equal than the specified interval. So, also ignore last one.
            records.loc[(records['intervals'].isnull()) | (records['intervals'] > 1), 'intervals'] = 0
            records[data_column] = records['intervals'] * pd.to_numeric(records[data_column])
            del records['intervals']
            assert old_num_records == records.shape[0]

        elif procedure.startswith('change_entries_to_true'):
            data_column = records.columns.tolist()[3]
            records[data_column] = True

        elif procedure.startswith('split_multi-coded'):
            data_column = records.columns.tolist()[3]
            records[data_column] = records[data_column].astype('str')
            num_new_records = 0
            for _, row in records.iterrows():
                num_new_records = num_new_records + len(row[data_column].split(', '))
            # Add entry for each code separated by ', '.
            # From: stackoverflow.com/questions/12680754/split-explode-pandas-dataframe-string-entry-to-separate-rows
            records = records.set_index(records.columns.drop(data_column, 1).tolist())[data_column]\
                             .str.split(', ', expand=True).stack().reset_index().rename(columns={0: data_column})\
                             .loc[:, records.columns]
            assert records.shape[0] == num_new_records

        else:
            raise Exception(f"Unexpected pre-processing {procedure}.")

    return records


def process_post_procedures(var_procedures, records):

    procedures = [] if not var_procedures else json.loads(var_procedures)

    for _, procedure in enumerate(procedures):
        # Apply all specified procedures after another.

        if procedure.startswith('add'):
            term = float(procedure.split('_')[1])
            data_column = records.columns.tolist()[3]
            records[data_column] = pd.to_numeric(records[data_column]) + term

        else:
            raise Exception(f"Unexpected post-processing {procedure}.")

    return records


def merge_item_to_variable(var_name, var_id, records):

    # Ensure proper index to allow iteration over it.
    records = records.reset_index(drop=True)

    if var_name == 'Body core temperature':
        core_temp_offset = {
            'Unknown': 0.0,
            'Tympanis': 0.4,
            'Oral': 0.5,
            'Achsel': 0.6,
            'Rektal': 0.0,
            'Kern': 0.0,
            'Leiste': 0.6,
            'vesikal': 0.0,
            'nasal': 0.5
        }
        synonyms = collections.OrderedDict([
            ('Unknown', ""),
            ('Tympanis', r"tympanis|ohr"),
            ('Oral', r"oral"),
            ('Achsel', r"achsel|achsilar|ax."),
            ('Rektal', r"rektal"),
            ('Kern', r"kern|picco"),
            ('Leiste', r"leiste|inguinal"),
            ('vesikal', r"vesikal|versikal"),
            ('nasal', r"nasal|nase"),
        ])
        regex_temp = r"\d\d(?:[.,]\d)?"
        records['patientid'] = records['patientid'].astype(np.int64)
        # 1. Correct location records of temperature
        # - Fix records with numericvalue 0 by converting them into valid numericvalue based on text or into temperature
        #   record.
        # - Copy records with location and temperature but no further location in textfield to final values, but keep
        #   also the original location entry.
        # - Mark remaining location records with itemid 0 so they come before other records when sorted by itemid later.
        converted_location_entries = 0
        for i, key in enumerate(synonyms.keys()):
            if synonyms[key]:
                records.loc[(records['itemid'] == 287) & (records['numericvalue'] == '0') &
                            (records['textvalue'].str.contains(synonyms[key], case=False, regex=True)),
                            ['numericvalue_text', 'numericvalue']] = [key, str(i)]
        for idx in records.loc[(records['itemid'] == 287) & (records['numericvalue'] == '0') &
                               (records['textvalue'].str.contains(regex_temp, regex=True))].index:
            temp = (re.findall(regex_temp, records.loc[idx, 'textvalue'])[0]).replace(',', '.')
            records.loc[idx, 'itemid'] = 81
            records.loc[idx, 'numericvalue'] = temp
            converted_location_entries += 1
        old_num_records = records.shape[0]
        records.drop(records[(records['itemid'] == 287) & (records['numericvalue'] == '0')].index, inplace=True)
        print(f"\tRemoved {old_num_records - records.shape[0]} temperature location records with code 0.")

        for idx in records.loc[(records['itemid'] == 287) & (records['textvalue'] != '\\N') &
                               (records['textvalue'].str.contains(regex_temp, regex=True))].index:
            # Check that textvalue contains no further location information.
            # Somehow a textvalue entry with \N passes the loop condition, so another test necessary here.
            low_location = str(records.loc[idx, 'textvalue'].lower())
            if not (any(len(re.findall(pat, low_location)) > 0 and pat for pat in synonyms.values())) and\
                    not ('ecmo' in low_location) and len(re.findall(regex_temp, low_location)) > 0:
                temp = float((re.findall(regex_temp, records.loc[idx, 'textvalue'])[0]).replace(',', '.'))
                records = records.append(pd.Series([records.loc[idx, 'patientid'], var_id,
                                                    records.loc[idx, 'displaytime'], '',
                                                    str(temp + core_temp_offset[records.loc[idx, 'numericvalue_text']]),
                                                    ''], index=records.columns), ignore_index=True)
                converted_location_entries += 1
        print(f"\tChanged {converted_location_entries} temperature location records to temperature recordings/values.")

        records.loc[records['itemid'] == 287, 'itemid'] = 0
        records['numericvalue'] = records['numericvalue'].astype(np.float32)

        # 2. Finalize remaining measurements (manual, CCO, T1, T2, T3, T4)
        # - Save entries with location entry but no temperature value in textvalue as final values.
        converted_temp_locations = 0
        for idx in records.loc[(records['itemid'] != 0) & (records['textvalue'] != '\\N') &
                               ~(records['textvalue'].str.contains(regex_temp, na=True, regex=True))].index:
            for i, key in enumerate(synonyms.keys()):
                if synonyms[key] and len(re.findall(synonyms[key], records.loc[idx, 'textvalue'].lower())) > 0:
                    temp = records.loc[idx, 'numericvalue'] + core_temp_offset[key]
                    records.loc[idx, 'numericvalue'] = temp
                    records.loc[idx, 'itemid'] = var_id
                    converted_temp_locations += 1
                    continue
        print(f"\tInferred location from text from {converted_temp_locations} temperature recordings.")

        # 3. All remaining entries (not itemid (var_id) yet) are considered as temperature recordings without location,
        # this location must be derived from the location records.
        # - Strip away unnecessary columns.
        # - Assign location to recordings succeeding a location entry until new patient or new location and no time
        #   limit reached. Perform this as numpy array for performance reasons.
        records.drop(['numericvalue_text', 'textvalue'], axis=1, inplace=True)

        old_num_records = records.shape[0]
        records.sort_values(['patientid', 'displaytime', 'itemid'], ascending=True, ignore_index=True, inplace=True)
        days_offset_bad_location = 7  # offset of days after which a location entry is not used for measurement anymore.
        curr_patient_id = 0
        curr_bp_loc = -1
        curr_bp_loc_date = -1
        records_array = records.to_numpy()
        for row in records_array:
            # entries: patientid, itemid, displaytime, data
            if row[0] != curr_patient_id:
                curr_patient_id = row[0]
                curr_bp_loc = -1
                curr_bp_loc_date = -1

            if row[1] == 0:
                # New location
                curr_bp_loc = int(row[3])
                curr_bp_loc_date = row[2]
            else:
                # Recording
                if curr_bp_loc != -1 and (row[2] - curr_bp_loc_date).days < days_offset_bad_location:
                    row[3] = row[3] + core_temp_offset[list(core_temp_offset.keys())[curr_bp_loc]]
                    row[1] = var_id
        records = pd.DataFrame(data=records_array, columns=records.columns)

        assert(old_num_records == records.shape[0])
        records.drop(records[records['itemid'] == 0].index, inplace=True)
        print(f"\tDiscarded {old_num_records - records.shape[0]} temperature location records.")
        old_num_records = records.shape[0]
        records.drop(records[records['itemid'] != var_id].index, inplace=True)
        print(f"\tRemoved {old_num_records - records.shape[0]} temperature records without location.")

    elif var_name == 'Measuring site body temperature':
        synonyms = collections.OrderedDict([
            ('Unknown', ""),
            ('Tympanis', r"tympanis|ohr"),
            ('Oral', r"oral"),
            ('Achsel', r"achsel|achsilar|ax."),
            ('Rektal', r"rektal"),
            ('Kern', r"kern|picco"),
            ('Leiste', r"leiste|inguinal"),
            ('vesikal', r"vesikal|versikal"),
            ('nasal', r"nasal|nase"),
        ])

        # 1. Fix wrong location entries with numericvalue 0 by converting them into valid numericvalue based on text.
        old_num_records_with_zero = records.loc[(records['itemid'] == 287) & (records['numericvalue'] == '0')].shape[0]
        for i, key in enumerate(synonyms.keys()):
            if synonyms[key]:
                records.loc[(records['itemid'] == 287) & (records['numericvalue'] == '0') &
                            (records['textvalue'].str.contains(synonyms[key], case=False, regex=True)),
                            ['numericvalue_text', 'numericvalue']] = [key, str(i)]
        entries_with_zero = records.loc[(records['itemid'] == 287) & (records['numericvalue'] == '0')].shape[0]
        print(f"\tConverted {old_num_records_with_zero - entries_with_zero} temperature location records with code 0.")
        old_num_records = records.shape[0]
        records.drop(records[(records['itemid'] == 287) & (records['numericvalue'] == '0')].index, inplace=True)
        print(f"\tRemoved {old_num_records - records.shape[0]} temperature location records with code 0.")

        # 2. Derive location from textvalue of temperature measurements.
        converted_temp_locations = 0
        for idx in records.loc[(records['itemid'] != 287) & (records['textvalue'] != '\\N')].index:
            for i, key in enumerate(synonyms.keys()):
                if synonyms[key] and len(re.findall(synonyms[key], records.loc[idx, 'textvalue'].lower())) > 0:
                    records.loc[idx, ['itemid', 'numericvalue_text', 'numericvalue']] = [287, key, str(i)]
                    converted_temp_locations += 1
                    continue
        print(f"\tInferred location from text from {converted_temp_locations} temperature recordings.")

        records.drop(['numericvalue', 'textvalue'], axis=1, inplace=True)
        records.drop(records[records['itemid'] != 287].index, inplace=True)

    elif var_name == 'Systolic blood pressure' or var_name == 'Diastolic blood pressure' or \
            var_name == 'Mean blood pressure':
        bp_offset = {
            47: {'arm': 7, 'thigh': -5},
            48: {'arm': 4, 'thigh': 11},
            49: {'arm': 3, 'thigh': 6}
        }

        # 1. Remove invalid non-inv BP locations.
        # (Don't parse location entries in textvalue of BP recordings, since only few and error-prone)
        old_num_records = records.shape[0]
        records['numericvalue'] = records['numericvalue'].astype(np.float32)
        records.drop(
            records[(records['itemid'] == 2653) & ~(records['numericvalue'].isin([1, 2, 3, 4, 5]))].index,
            inplace=True)
        print(f"\tRemoved {old_num_records - records.shape[0]} invalid BP location recordings.")
        records.loc[records['itemid'] == 2653, 'itemid'] = 0

        # 2. Remove BP measurements that contain comment that states an incorrect measurement.
        old_num_records = records.shape[0]
        records.drop(
            records[((records['itemid'] == 10) | (records['itemid'] == 11) | (records['itemid'] == 12) |
                     (records['itemid'] == 47) | (records['itemid'] == 48) | (records['itemid'] == 49)) &
                    records['textvalue'].str.contains(r"fehl|falsch", regex=True, case=False)].index, inplace=True)
        print(f"\tRemoved {old_num_records - records.shape[0]} incorrect BP recordings based on note (textvalue).")

        # 3. Correct non invasive BP recordings based on recent location entries.
        # - Strip away unnecessary columns.
        # - Assign location to recordings succeeding a location entry until new patient or new location and no time
        #   limit reached. Perform this as numpy array for performance reasons.
        records.drop(columns='textvalue', axis=1, inplace=True)

        old_num_records = records.shape[0]
        records.sort_values(['patientid', 'displaytime', 'itemid'], ascending=True, ignore_index=True, inplace=True)
        days_offset_bad_location = 7  # offset of days after which a location entry is not used for measurement anymore.
        curr_patient_id = 0
        curr_bp_loc = -1
        curr_bp_loc_date = -1
        records_array = records.to_numpy()
        for row in records_array:
            # entries: patientid, itemid, displaytime, data
            # Invasive BP recordings need no correction.
            if row[1] == 10 or row[1] == 11 or row[1] == 12:
                continue

            # Invasive femoral BPs also no correction.
            if row[1] == 2511 or row[1] == 2512 or row[1] == 2513:
                continue

            # Invasive brachial BPs of monitor LOINC code (MON_)
            if row[1] > 4000:
                continue

            if row[0] != curr_patient_id:
                curr_patient_id = row[0]
                curr_bp_loc = -1
                curr_bp_loc_date = -1

            if row[1] == 0:
                # New location
                curr_bp_loc = int(row[3])
                curr_bp_loc_date = row[2]
            else:
                # Recording
                if curr_bp_loc != -1 and (row[2] - curr_bp_loc_date).days < days_offset_bad_location:
                    row[3] = row[3] + bp_offset[row[1]]['thigh' if curr_bp_loc in [3, 4] else 'arm']
                else:
                    # Use arm offset as default.
                    row[3] = row[3] + bp_offset[row[1]]['arm']
        records = pd.DataFrame(data=records_array, columns=records.columns)

        assert(old_num_records == records.shape[0])
        records.drop(records[records['itemid'] == 0].index, inplace=True)
        print(f"\tDiscarded {old_num_records - records.shape[0]} BP location records.")

    elif var_name == 'Ventilation mode':
        # Use manual entries of ventilation mode (id 2213) and add entries for "assisted" whenever a positive automatic
        # minute volume is automatically transferred from a mechanical ventilator (id 321).
        ventilation_types = {
            "cppv": "controlled",
            "ippv": "controlled",
            "dk": "controlled",
            "simv": "assisted",
            "du_asb": "assisted",
            "cpap": "assisted",
            "spontan": "spontaneous",
            "hf_cpap": "assisted",
            "manuell": "manual",
            "standby": "manual",
            "hfv": "controlled",
            "cppv_asb": "controlled",
            "ippv_asb": "controlled",
            "simv_dk": "assisted",
            "simv_du": "assisted",
            "asb_cpap": "assisted",
            "simvcpap": "assisted",
            "manspont": "manual",
            "bipap": "assisted",
            "ni_bipap": "assisted",
            "blähen": "manual",
            "bipap_as": "assisted",
            "bipapass": "assisted",
            "s/t": "assisted",
            "duo_pap": "assisted",
            "asv": "controlled",
            "gali_spo": "spontaneous",
            "hfo2": "high-flow O2",
            "spont": "spontaneous",
            "cpap_asb": "assisted",
        }
        records = records.replace({'numericvalue_text': ventilation_types})

        # When duplicated time of recording (displaytime), use automatic recording since they are more reliable.
        old_num_records = records.shape[0]
        records.drop(records[(records['itemid'] == 321) & records['numericvalue'] <= 0].index, inplace=True)
        old_num_automatic_records = records.loc[(records['itemid'] == 321)].shape[0]
        records = records.sort_values(by=['patientid', 'itemid', 'displaytime'], ascending=True)
        records.drop_duplicates(subset=['patientid', 'displaytime'], keep='first', inplace=True)
        assert old_num_automatic_records == records.loc[(records['itemid'] == 321)].shape[0]
        records.loc[(records['itemid'] == 321) & records['numericvalue'] > 0, 'numericvalue_text'] = 'controlled'
        records.drop(columns='numericvalue', axis=1, inplace=True)
        records.loc[:, 'itemid'] = var_id

        # Transform categories into ordinals to use them as timeseries variable.
        ventilation_types_ordinals = {
            "spontaneous": 5,
            "high-flow O2": 4,
            "assisted": 3,
            "controlled": 2,
            "manual": 1
        }
        records = records.replace({'numericvalue_text': ventilation_types_ordinals})

        print(f"\tDiscarded {old_num_records - records.shape[0]} manual recorded ventilation types for automatic ones.")

    elif var_name == 'Is on automatic ventilation':
        # Use automatically transferred volumes from a mechanical ventilator (id 321) that should come every 15 min.
        assert all(records['itemid'] == 321)
        records['numericvalue'] = records['numericvalue'].astype(np.float32)
        records.drop(records[records['numericvalue'] <= 0].index, inplace=True)
        # Use an indicator variable to show that on mechanical ventilation.
        records['numericvalue'] = True

    elif var_name == 'Time on automatic ventilation':
        # Use automatically transferred volumes from a mechanical ventilator (id 321) that should come every 15 min.
        assert all(records['itemid'] == 321)
        records['numericvalue'] = records['numericvalue'].astype(np.float32)
        records.drop(records[records['numericvalue'] <= 0].index, inplace=True)
        # Each recording represents a time interval of 0.25h (here converted to days).
        records['numericvalue'] = (0.25 / 24)

    elif var_name == 'Tidal volume per body weight':
        # Use latest weight entry to normalize the tidal volume.
        assert all((records['itemid'] == 185) | (records['itemid'] == 962) | (records['itemid'] == 320))
        records['numericvalue'] = records['numericvalue'].astype(np.float32)
        records = filter_and_forward_fill_weight(records)
        records.drop(records[records['weight'].isna()].index, inplace=True)
        records['numericvalue'] = records['numericvalue']/records['weight']
        records.drop(columns='weight', axis=1, inplace=True)
        records.reset_index(drop=True, inplace=True)

    elif var_name == 'Creatinine clearance (urine)':
        # Use serum and urine creatinine to estimate creatinine clearance.
        # Blood and urine creatinine, hf to detect length of stay and urine outputs.
        assert all((records['itemid'] == 828) | (records['itemid'] == 888) | (records['itemid'] == 34) |
                   (records['itemid'] == 1206) | (records['itemid'] == 2247) | (records['itemid'] == 2248) |
                   (records['itemid'] == 2249) | (records['itemid'] == 2305))
        records['numericvalue'] = records['numericvalue'].astype(np.float32)

        # Summarize all urine items.
        records.loc[((records['itemid'] == 1206) | (records['itemid'] == 2247) |
                    (records['itemid'] == 2248) | (records['itemid'] == 2249) |
                    (records['itemid'] == 2305)), 'itemid'] = 2000
        # Remove invalid blood creatinine values.
        records.drop(records[(records['itemid'] == 828) & (records['numericvalue'] <= 0)].index, inplace=True)

        records = records.sort_values(by=['patientid', 'displaytime', 'itemid'], ascending=True)
        for idx, _ in records.loc[records['itemid'] == 888].iterrows():
            time = records.at[idx, 'displaytime']
            pat_24h_records = records.loc[(records['patientid'] == records.at[idx, 'patientid']) &
                                          ((records['displaytime'] <= time) &
                                           (records['displaytime'] >= time - dt.timedelta(hours=24)))]
            daily_blood_creatinine_values = pat_24h_records.loc[(records['itemid'] == 828), 'numericvalue']
            if daily_blood_creatinine_values.shape[0] > 0:
                blood_creatinine = daily_blood_creatinine_values.iloc[-1]
            else:
                continue  # No matching blood creatinine value detected.
            # Estimate duration at station. Normally every 15 minutes, so if for 22h assume 24h at station
            num_hf_recs = pat_24h_records.loc[(records['itemid'] == 34), 'numericvalue'].shape[0]
            if num_hf_recs >= 22 * 4:
                minutes_at_station = 24*60
            elif num_hf_recs > 0:
                minutes_at_station = num_hf_recs * 15
            else:
                continue
            daily_urine = pat_24h_records.loc[(records['itemid'] == 2000), 'numericvalue'].sum()

            records.loc[idx, 'numericvalue'] =\
                ((records.at[idx, 'numericvalue'] / blood_creatinine) * daily_urine) / minutes_at_station
            records.loc[idx, 'itemid'] = var_id
        records.drop(records[records['itemid'] != var_id].index, inplace=True)

    elif var_name == 'BMI':
        # Calculate BMI from two weight entries and the height.
        assert all((records['itemid'] == 185) | (records['itemid'] == 962) | (records['itemid'] == 186))
        records['numericvalue'] = records['numericvalue'].astype(np.float32)
        records.drop(records[((records['itemid'] == 185) | (records['itemid'] == 962)) &
                             (records['numericvalue'] < 30)].index, inplace=True)
        records.drop(records[(records['itemid'] == 186) & (records['numericvalue'] < 60)].index, inplace=True)
        # Iterate through all entries and when weight and height available for patient update entry accordingly.
        records = records.sort_values(by=['patientid', 'displaytime'], ascending=True)
        curr_patient_id = 0
        curr_weight = -1
        curr_height = -1
        for idx, record in records.iterrows():
            if record['patientid'] != curr_patient_id:
                curr_patient_id = record['patientid']
                curr_weight = -1
                curr_height = -1

            if record['itemid'] == 185 or record['itemid'] == 962:
                curr_weight = record['numericvalue']

            if record['itemid'] == 186:
                curr_height = record['numericvalue']

            if curr_weight != -1 and curr_height != -1:
                # If exactly the same treat as input error.
                if curr_weight == curr_height:
                    continue
                records.loc[idx, 'itemid'] = var_id
                records.loc[idx, 'numericvalue'] = curr_weight/((curr_height / 100) ** 2)

        records.drop(records[records['itemid'] != var_id].index, inplace=True)
        records.reset_index(drop=True, inplace=True)

    elif var_name == 'PAP - PEEP':
        # For each measurement of PAP (id 56), and PEEP (id 57) with less than 30 min in between calculate difference.
        assert all((records['itemid'] == 56) | (records['itemid'] == 57))
        records['numericvalue'] = records['numericvalue'].astype(np.float32)
        old_num_records = records.shape[0]
        records.drop(records[(records['itemid'] == 56) &
                             ((records['numericvalue'] <= 1) | (records['numericvalue'] > 80))].index, inplace=True)
        print(f"\tDiscarded {old_num_records - records.shape[0]} invalid PAP records [1,80].")
        old_num_records = records.shape[0]
        records.drop(records[(records['itemid'] == 57) &
                             ((records['numericvalue'] <= 1) | (records['numericvalue'] > 40))].index, inplace=True)
        print(f"\tDiscarded {old_num_records - records.shape[0]} invalid PEEP records [1,40].")

        records.sort_values(['patientid', 'displaytime', 'itemid'], ascending=True, ignore_index=True, inplace=True)
        old_num_records = records.shape[0]
        offset_interval_hours = 0.5
        curr_patient_id = -1
        curr_pap_time = -1
        curr_pap = -1
        curr_peep_time = -1
        curr_peep = -1
        records_array = records.to_numpy()
        for row in records_array:
            # entries: patientid, itemid, displaytime, data
            if row[0] != curr_patient_id:
                # New patient, so not both values available yet
                curr_patient_id = row[0]
                curr_pap_time = -1
                curr_pap = -1
                curr_peep_time = -1
                curr_peep = -1

            if row[1] == 56:
                curr_pap_time = row[2]
                curr_pap = row[3]

            if row[1] == 57:
                curr_peep_time = row[2]
                curr_peep = row[3]

            if curr_pap != -1 and curr_peep != -1 and \
                    ((curr_pap_time - curr_peep_time).total_seconds())/3600 <= offset_interval_hours:
                row[1] = var_id
                row[3] = curr_pap - curr_peep

        records = pd.DataFrame(data=records_array, columns=records.columns)
        assert(old_num_records == records.shape[0])
        records.drop(records[records['itemid'] != var_id].index, inplace=True)
        print(f"\tDiscarded {old_num_records - records.shape[0]} entries.")
        assert all(records['itemid'] == var_id)

    elif var_name == 'eGFR':
        # For each Creatinine (id 828) recording, and existing Gender (m/f = 1/2) (id 128) & Age (id 96) calculate eGFR.
        # Use: https://www.kidney.org/content/ckd-epi-creatinine-equation-2009
        # No information about Black people, so ignore this part of the formula.
        assert all((records['itemid'] == 828) | (records['itemid'] == 128) | (records['itemid'] == 96))
        records['numericvalue'] = records['numericvalue'].astype(np.float32)
        old_num_records = records.shape[0]
        records.drop(records[(records['itemid'] == 828) &
                             ((records['numericvalue'] <= 0) | (records['numericvalue'] > 500))].index, inplace=True)
        print(f"\tDiscarded {old_num_records - records.shape[0]} invalid Creatinine records [0,500].")
        old_num_records = records.shape[0]
        records.drop(records[(records['itemid'] == 96) &
                             ((records['numericvalue'] <= 17) | (records['numericvalue'] > 120))].index, inplace=True)
        print(f"\tDiscarded {old_num_records - records.shape[0]} invalid Age records [18,120].")

        records.sort_values(['patientid', 'displaytime', 'itemid'], ascending=True, ignore_index=True, inplace=True)
        old_num_records = records.shape[0]
        curr_patient_id = -1
        curr_age = -1
        curr_gender = -1
        records_array = records.to_numpy()
        for row in records_array:
            # entries: patientid, itemid, displaytime, data
            if row[0] != curr_patient_id:
                # New patient, so not both values available yet
                curr_patient_id = row[0]
                curr_age = -1
                curr_gender = -1

            if row[1] == 128 and (row[3] == 1 or row[3] == 2):
                curr_gender = row[3]

            if row[1] == 96:
                curr_age = row[3]

            if row[1] == 828 and curr_gender != -1 and curr_age != -1:
                creatinine = row[3]
                kappa = 0.9 if curr_gender == 1 else 0.7
                alpha = -0.411 if curr_gender == 1 else -0.329
                normalizing = 1 if curr_gender == 1 else 1.018
                row[3] = 141 * (min(creatinine / kappa, 1)**alpha) * (max(creatinine / kappa, 1)**(-1.209)) * \
                    (0.993**curr_age) * normalizing
                row[1] = var_id

        records = pd.DataFrame(data=records_array, columns=records.columns)
        assert(old_num_records == records.shape[0])
        records.drop(records[records['itemid'] != var_id].index, inplace=True)
        print(f"\tDiscarded {old_num_records - records.shape[0]} entries.")
        assert all(records['itemid'] == var_id)

    elif var_name == 'paO2/FiO2':
        # For each measurement of pO2 (id 341), use the latest FiO2 (id 29) that is not older than 24h.
        records['numericvalue'] = records['numericvalue'].astype(np.float32)
        old_num_records = records.shape[0]
        records.drop(records[(records['itemid'] == 341) &
                             ((records['numericvalue'] <= 1) | (records['numericvalue'] > 760))].index, inplace=True)
        print(f"\tDiscarded {old_num_records - records.shape[0]} invalid pO2 records [1,760].")
        old_num_records = records.shape[0]
        records.drop(records[(records['itemid'] == 29) &
                             ((records['numericvalue'] <= 0) | (records['numericvalue'] > 100))].index, inplace=True)
        print(f"\tDiscarded {old_num_records - records.shape[0]} invalid Fi02 records [1,100].")

        # Alternative code if no offset necessary
        # records['pO2'] = records.loc[((records['itemid'] == 341) | (records['itemid'] == 2457)), 'numericvalue']
        # records['fiO2'] = records.loc[((records['itemid'] == 29) | (records['itemid'] == 2356)), 'numericvalue']
        # records.sort_values(['patientid', 'displaytime', 'itemid'], ascending=False, ignore_index=True, inplace=True)
        # records[['pO2', 'fiO2']] = records.groupby('patientid')['pO2', 'fiO2'].backfill()

        records.sort_values(['patientid', 'displaytime', 'itemid'], ascending=True, ignore_index=True, inplace=True)
        old_num_records = records.shape[0]
        offset_interval_hours = 24
        curr_patient_id = -1
        curr_fio2_time = -1
        curr_fio2 = -1
        records_array = records.to_numpy()
        for row in records_array:
            # entries: patientid, itemid, displaytime, data
            if row[0] != curr_patient_id:
                # New patient, so not both values available yet
                curr_patient_id = row[0]
                curr_fio2_time = -1
                curr_fio2 = -1

            if row[1] == 29:
                curr_fio2_time = row[2]
                curr_fio2 = row[3]

            if (row[1] == 341) and (curr_fio2 != -1):
                curr_po2 = row[3]
                curr_po2_time = row[2]
                if ((curr_po2_time - curr_fio2_time).total_seconds())/3600 <= offset_interval_hours:
                    row[1] = var_id
                    row[3] = curr_po2 / (curr_fio2 / 100.)

        records = pd.DataFrame(data=records_array, columns=records.columns)
        assert(old_num_records == records.shape[0])
        records.drop(records[records['itemid'] == 29].index, inplace=True)
        print(f"\tDiscarded {old_num_records - records.shape[0]} FiO2 entries (only ratio for pO2 entries).")
        old_num_records = records.shape[0]
        records.drop(records[records['itemid'] == 341].index, inplace=True)
        print(f"\tDiscarded {old_num_records - records.shape[0]} pO2 entries where no ratio could be determined.")
        assert all(records['itemid'] == var_id)

    elif var_name == 'GCS score':
        # For each measurement of GCS motor (id 3430), GCS verbal (id 3431), GCS eye (id 3432) determine GCS score.
        records['numericvalue'] = records['numericvalue'].astype(np.int)
        old_num_records = records.shape[0]
        records.drop(records[(records['numericvalue'] < 1)].index, inplace=True)
        records.drop(records[(records['itemid'] == 3430) & (records['numericvalue'] > 6)].index, inplace=True)
        records.drop(records[(records['itemid'] == 3431) & (records['numericvalue'] > 5)].index, inplace=True)
        records.drop(records[(records['itemid'] == 3432) & (records['numericvalue'] > 4)].index, inplace=True)
        print(f"\tDiscarded {old_num_records - records.shape[0]} invalid GCS scores.")
        old_num_records = records.shape[0]

        records['itemid'] = records['itemid'].map({3430: 'GCS motor', 3431: 'GCS verbal', 3432: 'GCS eye'})
        records = records.pivot_table(index=['patientid', 'displaytime'], columns='itemid', values='numericvalue',
                                      aggfunc=np.min)
        removed_duplicates = old_num_records - records.count().sum()
        print(f"\tDiscarded {removed_duplicates} duplicated GCS records during pivoting (keep min value).")
        records = records.reset_index()
        # Count the number of GCS recordings that are in an incomplete row.
        not_complete = records.loc[records['GCS eye'].isnull() | records['GCS motor'].isnull() |
                                   records['GCS verbal'].isnull(), ['GCS motor', 'GCS eye', 'GCS verbal']].count().sum()
        records.drop(records.loc[records['GCS eye'].isnull() | records['GCS motor'].isnull() |
                                 records['GCS verbal'].isnull()].index, inplace=True)
        assert records.shape[0] * 3 + not_complete + removed_duplicates == old_num_records
        print(f"\tDiscarded {not_complete} GCS records since not all three present.")

        records['itemid'] = var_id
        records['GCS score'] = records['GCS eye'] + records['GCS motor'] + records['GCS verbal']
        records.drop(columns=['GCS eye', 'GCS motor', 'GCS verbal'], axis=1, inplace=True)

    elif var_name == 'Admission reason':
        # Use diagnosis at admission for score calculation to distinguish (non-)surgical patients.
        assert all((records['itemid'] == 3495) | (records['itemid'] == 3496))
        records.drop(records[records['numericvalue_text'] == '0'].index, inplace=True)
        admission_mapping = \
            {"Chron_HK": "Cardiovascular disease",
             "Herzklap": "Cardiovascular disease",
             "HerzK": "Cardiovascular disease",
             "GefäßChi": "Cardiovascular disease",
             "CHF": "Cardiovascular disease",
             "KHK": "Cardiovascular disease",
             "KardVasc": "Cardiovascular disease",
             "HRSt": "Cardiovascular disease",
             "kardScho": "Cardiovascular disease",
             "LungEmbo": "Cardiovascular disease",
             "LungÖdem": "Cardiovascular disease",
             "Herzstil": "Cardiovascular disease",
             "AoAneur": "Cardiovascular disease",
             "AHT": "Cardiovascular disease",
             "Gastro": "GI disorder",
             "GIPerfOb": "GI disorder",
             "GI_Blut": "GI disorder",
             "GastroIn": "GI disorder",
             "Laminekt": "Neurological disorder",
             "Neuro": "Neurological disorder",
             "CraniNeo": "Neurological disorder",
             "CraniHäm": "Neurological disorder",
             "SHT": "Neurological disorder",
             "Krampf": "Neurological disorder",
             "TrauSchä": "Neurological disorder",
             "Resp": "Respiratory disorder",
             "RespIns": "Respiratory disorder",
             "COPD": "Respiratory disorder",
             "AtemStil": "Respiratory disorder",
             "Atemstil": "Respiratory disorder",
             "AsthmaAl": "Respiratory disorder",
             "POpBlutu": "Trauma or bleeding",
             "BlutungS": "Trauma or bleeding",
             "Blutung": "Trauma or bleeding",
             "TrauMult": "Trauma or bleeding",
             "ThorxNeo": "Neoplasm",
             "GIOpNe": "Neoplasm",
             "NierNeo": "Neoplasm",
             "Neoplasm": "Neoplasm",
             "Seps_po": "Sepsis or infection",
             "Sepsis": "Sepsis or infection",
             "Infekt": "Sepsis or infection",
             "MetRenal": "Other",
             "Niere": "Other",
             "AspirPoi": "Other",
             "DiabKeto": "Other",
             "NTP": "Other",
             "Überdos": "Other"}
        records = records.replace({'numericvalue_text': admission_mapping})
        assert not records['numericvalue_text'].isnull().any()
        # Keep more specific categories.
        records['numericvalue_text'] = pd.Categorical(
            records['numericvalue_text'], ["Sepsis or infection", "Trauma or bleeding", "Neoplasm",
                                           "Respiratory disorder", "Neurological disorder", "GI disorder",
                                           "Cardiovascular disease", "Other"])
        records = records.sort_values('numericvalue_text')
        records.drop_duplicates(subset=['patientid', 'displaytime'], keep='first', inplace=True,
                                ignore_index=True)
        records['numericvalue_text'] = records['numericvalue_text'].astype(str)

    elif var_name == 'Has decubitus':
        # Use recordings of at least decubitus stage 1 (value 0) as indicator.
        records['numericvalue'] = records['numericvalue'].astype(np.int)
        records.drop(records[(records['numericvalue'] < 0)].index, inplace=True)
        records.loc[:, 'numericvalue'] = True
        records.drop_duplicates(subset=['patientid', 'displaytime'], keep='first', inplace=True, ignore_index=True)

    elif var_name == 'Antithrombotic agents therapeutic dosage' or\
            var_name == 'Antithrombotic agents prophylactic dosage':
        # Transform antithrombotic agents into prophylactic and therapeutic dosages.
        records['numericvalue'] = records['numericvalue'].astype(np.float32)
        assert all((records['itemid'] == 1510) | (records['itemid'] == 1577) | (records['itemid'] == 1578) |
                   (records['itemid'] == 1628) | (records['itemid'] == 1629) | (records['itemid'] == 2654) |
                   (records['itemid'] == 3936) | (records['itemid'] == 4254) | (records['itemid'] == 4573) |
                   (records['itemid'] == 5652) | (records['itemid'] == 5653) | (records['itemid'] == 5927))
        records['low-dose'] = False
        records['high-dose'] = False
        records.loc[((records['itemid'] == 1510) & (0 < records['numericvalue']) & (records['numericvalue'] <= 40)) |
                    ((records['itemid'] == 1577) & (0 < records['numericvalue']) & (records['numericvalue'] <= 500)) |
                    (records['itemid'] == 1578) |
                    (records['itemid'] == 1628) |
                    ((records['itemid'] == 4573) & (0 < records['numericvalue']) & (records['numericvalue'] <= 2.5)),
                    'low-dose'] = True
        records.loc[((records['itemid'] == 1510) & (records['numericvalue'] > 40)) |
                    ((records['itemid'] == 1577) & (500 < records['numericvalue']) & (records['numericvalue'] <= 75000))
                    | (records['itemid'] == 1629) |
                    (records['itemid'] == 2654) |
                    (records['itemid'] == 3936) |
                    (records['itemid'] == 4254) |
                    ((records['itemid'] == 4573) & (2.5 < records['numericvalue'])) |
                    (records['itemid'] == 5652) |
                    (records['itemid'] == 5653) |
                    (records['itemid'] == 5927),
                    'high-dose'] = True
        records.drop(records[~(records['low-dose'] | records['high-dose'])].index, inplace=True)
        assert not any((records['low-dose']) & (records['high-dose']))
        # If low-dose and high-dose at same time, only retain last.
        records.sort_values(['patientid', 'displaytime', 'high-dose'], ascending=[True, True, True], inplace=True,
                            ignore_index=True)
        old_num_records = records.shape[0]
        records.drop_duplicates(subset=['patientid', 'displaytime'], keep='last', inplace=True, ignore_index=True)
        print(f"\tDiscarded {old_num_records - records.shape[0]} records collect at same time, prefer high-dose.")
        if var_name == 'Antithrombotic agents therapeutic dosage':
            records.drop(records[~(records['high-dose'])].index, inplace=True)  # 'therapeutic'
        if var_name == 'Antithrombotic agents prophylactic dosage':
            records.drop(records[~(records['low-dose'])].index, inplace=True)  # 'prophylactic'
        records['numericvalue'] = True
        records.drop(columns=['low-dose', 'high-dose'], axis=1, inplace=True)

    elif var_name == 'Cardiac stimulants (epinephrine equivalence dosage)' or\
            var_name == 'Norepinephrine and Dopamine (norepinephrine equivalence dosage)':
        records['numericvalue'] = records['numericvalue'].astype(np.float32)
        if var_name == 'Cardiac stimulants (epinephrine equivalence dosage)':
            assert all((records['itemid'] == 1534) | (records['itemid'] == 1667) | (records['itemid'] == 1668) |
                       (records['itemid'] == 1734) | (records['itemid'] == 4365) |
                       (records['itemid'] == 185) | (records['itemid'] == 962))
        if var_name == 'Norepinephrine and Dopamine (norepinephrine equivalence dosage)':
            assert all((records['itemid'] == 1469) | (records['itemid'] == 1538) | (records['itemid'] == 1728) |
                       (records['itemid'] == 185) | (records['itemid'] == 962))
        # Remove artifacts.
        old_num_records = records.shape[0]
        records.drop(records[((records['itemid'] == 1534) & (records['numericvalue'] > 20)) |
                             ((records['itemid'] == 1667) & (records['numericvalue'] > 30)) |
                             ((records['itemid'] == 1668) & (records['numericvalue'] > 5)) |
                             ((records['itemid'] == 1734) & (records['numericvalue'] > 0.75)) |
                             ((records['itemid'] == 4365) & (records['numericvalue'] > 12.7)) |
                             ((records['itemid'] == 1469) & (records['numericvalue'] > 5)) |
                             ((records['itemid'] == 1538) & (records['numericvalue'] > 10)) |
                             ((records['itemid'] == 1728) & (records['numericvalue'] > 5))].index, inplace=True)
        print(f"\tDiscarded {old_num_records - records.shape[0]} records with too high values.")
        old_num_records = records.shape[0]
        records.drop(records[(records['numericvalue'] <= 0)].index, inplace=True)
        print(f"\tDiscarded {old_num_records - records.shape[0]} records with zero values.")

        # Determine the weight of the patient or use average weight as fallback.
        average_weight = 78
        records['numericvalue'] = records['numericvalue'].astype(np.float32)
        records = filter_and_forward_fill_weight(records)
        records.loc[records['weight'].isna(), 'numericvalue'] = average_weight

        # Transform all entries to mg.
        # 1469, 1534, 1538, 1668, 1734: Transform mug/kg/min into mg.
        per_kg_min_mask = ((records['itemid'] == 1469) | (records['itemid'] == 1534) | (records['itemid'] == 1538) |
                           (records['itemid'] == 1668) | (records['itemid'] == 1734))
        # All except 1538 (1h interval) given in 20 min interval.
        records.loc[per_kg_min_mask, 'numericvalue'] = \
            records.loc[per_kg_min_mask, 'numericvalue'] * records.loc[per_kg_min_mask, 'weight'] * 20
        records.loc[(records['itemid'] == 1538), 'numericvalue'] = \
            records.loc[(records['itemid'] == 1538), 'numericvalue'] * 3
        # 4365 given in mg/d also transform to mg. Interval is 20 min.
        records.loc[records['itemid'] == 4365, 'numericvalue'] =\
            records.loc[records['itemid'] == 4365, 'numericvalue'] * (20. / (24 * 60))

        # Change into equivalence dosages. Use DDD from WHO
        # 'Cardiac stimulants'
        # Dobutamin
        records.loc[records['itemid'] == 1534, 'numericvalue'] =\
            records.loc[records['itemid'] == 1534, 'numericvalue'] * (0.5 / 500)
        # Milrinon
        records.loc[records['itemid'] == 1734, 'numericvalue'] = \
            records.loc[records['itemid'] == 1734, 'numericvalue'] * (0.5 / 50)
        # Levosimendan
        records.loc[records['itemid'] == 4365, 'numericvalue'] = \
            records.loc[records['itemid'] == 4365, 'numericvalue'] * (0.5 / 11)

        # 'Norepinephrine and Dopamine':
        # Dopamine
        records.loc[records['itemid'] == 1538, 'numericvalue'] =\
            records.loc[records['itemid'] == 1538, 'numericvalue'] * (6 / 500)

        # Sum duplicates to prevent that they are removed.
        records['numericvalue'] = records.groupby(['patientid', 'displaytime'])['numericvalue'].transform('sum')
        old_num_records = records.shape[0]
        records.drop_duplicates(subset=['patientid', 'displaytime'], keep='last', inplace=True, ignore_index=True)
        print(f"\tDiscarded {old_num_records - records.shape[0]} records collected at same time, summed dosages.")
        records.drop(columns='weight', axis=1, inplace=True)
        records.reset_index(drop=True, inplace=True)

    elif var_name == 'Glucocorticoids (cortison equivalence dosage)':
        assert all((records['itemid'] == 1501) | (records['itemid'] == 1522) | (records['itemid'] == 1523) |
                   (records['itemid'] == 1556) | (records['itemid'] == 1583) | (records['itemid'] == 1584) |
                   (records['itemid'] == 1659))
        records['numericvalue'] = records['numericvalue'].astype(np.float32)
        # Remove artifacts.
        old_num_records = records.shape[0]
        records.drop(records[((records['itemid'] == 1501) & (records['numericvalue'] > 3000)) |
                             ((records['itemid'] == 1522) & (records['numericvalue'] > 1000)) |
                             ((records['itemid'] == 1523) & (records['numericvalue'] > 1000)) |
                             ((records['itemid'] == 1556) & (records['numericvalue'] > 250)) |
                             ((records['itemid'] == 1583) & (records['numericvalue'] > 500)) |
                             ((records['itemid'] == 1584) & (records['numericvalue'] > 200)) |
                             ((records['itemid'] == 1659) & (records['numericvalue'] > 1000))].index, inplace=True)
        print(f"\tDiscarded {old_num_records - records.shape[0]} records with too high values.")
        old_num_records = records.shape[0]
        records.drop(records[(records['numericvalue'] <= 0)].index, inplace=True)
        print(f"\tDiscarded {old_num_records - records.shape[0]} records with zero values.")
        # Change perfusor value (mg/d) into mg, assume an update interval of 20 min.
        records.loc[records['itemid'] == 1583, 'numericvalue'] =\
            records.loc[records['itemid'] == 1583, 'numericvalue'] * (20. / (24 * 60))
        # Change into equivalence dosages.
        records.loc[records['itemid'] == 1501, 'numericvalue'] =\
            records.loc[records['itemid'] == 1501, 'numericvalue'] * 5
        records.loc[records['itemid'] == 1522, 'numericvalue'] = \
            records.loc[records['itemid'] == 1522, 'numericvalue'] * 4
        records.loc[records['itemid'] == 1523, 'numericvalue'] = \
            records.loc[records['itemid'] == 1523, 'numericvalue'] * 4
        records.loc[records['itemid'] == 1556, 'numericvalue'] = \
            records.loc[records['itemid'] == 1556, 'numericvalue'] * 20
        records.loc[records['itemid'] == 1659, 'numericvalue'] = \
            records.loc[records['itemid'] == 1659, 'numericvalue'] * 4
        # Sum duplicates to prevent that they are removed.
        records['numericvalue'] = records.groupby(['patientid', 'displaytime'])['numericvalue'].transform('sum')
        old_num_records = records.shape[0]
        records.drop_duplicates(subset=['patientid', 'displaytime'], keep='last', inplace=True, ignore_index=True)
        print(f"\tDiscarded {old_num_records - records.shape[0]} records collected at same time, summed dosages.")

    else:
        # Default: reduce to four columns and keep name of fourth column.
        records = records[records.columns[0:4]]

    # Assign merged variable id.
    records['itemid'] = var_id
    return records


def filter_and_forward_fill_weight(records):
    # Forward fill the weight items 185, 962 into new column weight. Only consider weights over 20 as for weight item.
    records['weight'] = records.loc[((records['itemid'] == 185) | (records['itemid'] == 962)) &
                                    (records['numericvalue'] >= 20), 'numericvalue']
    records = records.sort_values(by=['patientid', 'displaytime'], ascending=True)
    records['displaytime_index'] = records['displaytime']
    records = records.set_index('displaytime_index')
    records['weight'] = records.groupby('patientid')['weight'].transform(lambda v: v.ffill())
    records.drop(records[(records['itemid'] == 185) | (records['itemid'] == 962)].index, inplace=True)
    records.reset_index(drop=True, inplace=True)
    return records


if __name__ == '__main__':
    main()
