"""
Script to add missing values for few static variables.
"""
import argparse
from research_database.research_database_communication import ResearchDBConnection
from helper.util import get_pid_case_ids_map

import pandas as pd
import datetime as dt


def main():
    parser = argparse.ArgumentParser(description='Add missing values for selected variables.')
    parser.add_argument('password', type=str, help='Database password.')
    parser.add_argument('--variable_names', nargs='+', help='Only convert specified variables.', default='')
    args, _ = parser.parse_known_args()

    db_conn = ResearchDBConnection(args.password)
    stays = db_conn.read_table_from_db('stays')
    stays_pid_case_ids = get_pid_case_ids_map(db_conn, stays)
    all_cases_of_all_patients = [item for sublist in list(stays_pid_case_ids.values()) for item in sublist]
    incl_cases = set(stays['fallnummer'].tolist())

    var_name = 'Patient class'
    var_type = 'categorical'
    default_value = 'Inpatient'
    stays_wout_value_per_hosp_stay = db_conn.get_stays_without_value_per_hosp_stay(var_name)
    additional_values = pd.DataFrame({'patientid': [], 'displaytime': [],
                                      'textvalue' if var_type == 'categorical' else 'numericvalue': []})
    for idx, stay in stays_wout_value_per_hosp_stay.iterrows():
        additional_values = additional_values.append(pd.Series([stay['fallnummer'], stay['intbis'], default_value],
                                                     index=additional_values.columns), ignore_index=True)
    print(f"{var_name}: Added {additional_values.shape[0]} additional value(s).")
    db_conn.write_values_to_db(var_name, additional_values)

    var_name = 'Gender'
    var_type = 'categorical'
    # Only single instance with a value that is recorded later, re-use it.
    additional_values = pd.DataFrame({'patientid': [], 'displaytime': [],
                                      'textvalue' if var_type == 'categorical' else 'numericvalue': []})
    additional_values = additional_values.append(pd.Series(['55660000', dt.datetime(2014, 10, 15, 16, 30), 'Male'],
                                                 index=additional_values.columns), ignore_index=True)
    print(f"{var_name}: Added {additional_values.shape[0]} additional value(s).")
    db_conn.write_values_to_db(var_name, additional_values)

    var_name = 'Responsible clinic'
    var_type = 'categorical'
    default_value = 'Other'
    # Fill all missing values with super category "Other".
    stays_wout_value_per_hosp_stay = db_conn.get_stays_without_value_per_hosp_stay(var_name)
    additional_values = pd.DataFrame({'patientid': [], 'displaytime': [],
                                      'textvalue' if var_type == 'categorical' else 'numericvalue': []})
    for idx, stay in stays_wout_value_per_hosp_stay.iterrows():
        additional_values = additional_values.append(pd.Series([stay['fallnummer'], stay['intbis'], default_value],
                                                     index=additional_values.columns), ignore_index=True)
    print(f"{var_name}: Added {additional_values.shape[0]} additional value(s).")
    db_conn.write_values_to_db(var_name, additional_values)


if __name__ == '__main__':
    main()
