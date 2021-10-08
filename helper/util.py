"""
Collection of general helper functions.
"""

import functools
import traceback
import re
from string import Template

import pandas as pd
import datetime


def get_pid_case_ids_map(db_conn, stays):
    """
    Get all cases associated with a pid.
    """
    transfers = db_conn.read_table_from_db('transfers')
    return transfers.loc[transfers['patientid'].isin(stays['patientid']), ['patientid', 'fallnummer']]\
        .drop_duplicates().groupby('patientid')['fallnummer'].apply(list).to_dict()


def output_wrapper(func):
    """
    Taken from https://stackoverflow.com/questions/44488645/python-multiprocessing-subprocesses-with-ordered-printing
    """
    @functools.wraps(func)  # we need this to unravel the target function name
    def decorated(*args, **kwargs):  # and now for the wrapper, nothing new here
        from io import StringIO
        import sys
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout, sys.stderr = StringIO(), StringIO()  # use our buffers instead
        result = None  # in case a call fails
        try:
            result = func(*args, **kwargs)  # call our wrapped process function
        except Exception as e:  # too broad but good enough as an example
            print(e)  # NOTE: the exception is also printed to the captured STDOUT
            print(traceback.print_exc())
        # rewind our buffers:
        sys.stdout.seek(0)
        sys.stderr.seek(0)
        # reset outputs streams and print stored outputs:
        stdout = sys.stdout.getvalue()
        stderr = sys.stderr.getvalue()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        if stdout:
            print(stdout, end='')
        if stderr:
            print(stderr, end='')
        return result
    return decorated


def parse_time(time_str):
    # From: https://stackoverflow.com/questions/4628122/how-to-construct-a-timedelta-object-from-a-simple-string
    regex = re.compile(r'((?P<days>\d+?)d)?((?P<hours>\d+?)h)?((?P<minutes>\d+?)m)?')
    parts = regex.match(time_str)
    if not parts:
        raise Exception('Bad time span formatting in feature method.')
    parts = parts.groupdict()
    time_params = {}
    for (name, param) in parts.items():
        if param:
            time_params[name] = int(param)
    return datetime.timedelta(**time_params)


def format_timedelta(timedelta_value):
    # From: https://stackoverflow.com/questions/8906926/formatting-timedelta-objects
    d = {"days": timedelta_value.days}
    d["hours"], rem = divmod(timedelta_value.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    if d["seconds"] != 0:
        if d["days"] != 0:
            fmt = "{days}d {hours}:{minutes}:{seconds}"
        else:
            fmt = "{hours}:{minutes}:{seconds}"
    elif d["minutes"] != 0:
        if d["days"] != 0:
            fmt = "{days}d {hours}:{minutes}"
        else:
            fmt = "{hours}:{minutes}"
    elif d["hours"] != 0:
        if d["days"] != 0:
            fmt = "{days}d {hours}h"
        else:
            fmt = "{hours}h"
    else:
        fmt = "{days}d"
    return fmt.format(**d)


class DeltaTemplate(Template):
    delimiter = "%"


def strfdelta(td, fmt):
    # From: https://stackoverflow.com/questions/8906926/formatting-timedelta-objects
    d = {"D": td.days}
    hours, rem = divmod(td.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d["H"] = '{:02d}'.format(hours)
    d["M"] = '{:02d}'.format(minutes)
    d["S"] = '{:02d}'.format(seconds)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)


def format_timedelta_opt_days(td):
    if td.days > 0:
        return strfdelta(td, '%D d %H:%M:%S')
    else:
        return strfdelta(td, '%H:%M:%S')


def read_features_years_labels(data_csv):
    """
    Function to read and correctly format the output of the preprocessing with features, labels, years.
    """
    # 1. Read csv file.
    data = pd.read_csv(data_csv, low_memory=False)

    # 2. Define columns.
    # Data is supposed to contain two non-feature columns. Remove them for feature selection and then add them again.
    year_column = 'year_icu_out'
    label_column = 'label'
    # TODO: For debugging
    # data = filter_for_saps_and_static_feat(data, year_column, label_column)
    data[year_column] = pd.to_datetime(data[year_column])
    data[label_column] = data[label_column].astype(int)
    feature_columns = [col for col in data.columns if col not in [year_column, label_column]]

    # 3. Cast columns to correct data types.
    numerical_columns, nominal_columns = derive_column_types(data.loc[:, feature_columns])
    data[numerical_columns] = data[numerical_columns].apply(pd.to_numeric)
    data[nominal_columns] = data[nominal_columns].astype(str)

    print(f"Read feature, label, year csv input with {data.shape[0]} entries and {len(feature_columns)} features.")

    return feature_columns, year_column, label_column, data


def filter_for_saps_and_static_feat(data, year_column, label_column):
    included_columns = ['Age (last all data)',
                        'Hematologic malignancy (last all data)',
                        'Metastatic cancer (last all data)',
                        'AIDS (last all data)',
                        'Type of admission (last per hospital stay)',
                        'Bilirubin total (median per icu stay)',
                        'GCS score (min 1d)',
                        'Blood Urea Nitrogen (max 1d)',
                        'Leucocytes (min 1d)',
                        'Leucocytes (max 1d)',
                        'pO2 (min 1d)',
                        'Potassium (max 1d)',
                        'Bicarbonate (min 1d)',
                        'Sodium (min 1d)',
                        'Potassium (min 1d)',
                        'Sodium (max 1d)',
                        'Urine volume out (extrapolate 1d)',
                        'Body core temperature (max 1d)',
                        'Ventilation mode (min 1h)',
                        'Systolic blood pressure (max 1d)',
                        'Systolic blood pressure (min 1d)',
                        'Heart rate (max 1d)',
                        'Heart rate (min 1d)',
                        'Is 3-day ICU readmission',
                        'ICU station',
                        'ICU length of stay [days]',
                        'Length of stay before ICU [days]',
                        year_column,
                        label_column]
    data.drop(data.columns.difference(included_columns), axis=1, inplace=True)
    return data


def create_splits(data, feature_columns, year_column, label_column):
    """
    Create a dict of dict of trained models for all splits and train/test/valid sets.
    """
    # Create dataframe with row for each split.
    splits = pd.DataFrame({'train_year_start': [], 'train_year_end': [], 'valid_year': [], 'test_year': [], 'full': []})
    # There are 5 temporal splits: 2017/2016/2015/2014/2013 + 7 years (train/valid).
    test_split_years = [2013, 2014, 2015, 2016, 2017]
    for test_year in test_split_years:
        splits = splits.append({'train_year_start': test_year - 7,
                                'train_year_end': test_year - 2,
                                'valid_year': test_year - 1,
                                'test_year': test_year}, ignore_index=True)
    # There is one split for final testing: 2019 (test), 2018 (valid), 2017 - 2006 (training)
    # For experiments use only: 2018 (test), 2017 (valid), 2016 - 2006 (training)
    splits = splits.append({'train_year_start': 2006,
                            'train_year_end': 2017,
                            'valid_year': 2018,
                            'test_year': 2019}, ignore_index=True)

    for idx, split in splits.iterrows():
        x_train, y_train, x_valid, y_valid, x_test, y_test = get_split(split, data, feature_columns, year_column,
                                                                       label_column)
        splits.loc[idx, 'train_n'] = x_train.shape[0]
        splits.loc[idx, 'valid_n'] = x_valid.shape[0]
        splits.loc[idx, 'test_n'] = x_test.shape[0]

    return splits


def get_split(split, data, feature_columns, year_column, label_column):
    test_upper_bound = datetime.datetime(int(split['test_year']) + 1, 1, 1)
    valid_upper_bound = datetime.datetime(int(split['valid_year']) + 1, 1, 1)
    train_upper_bound = datetime.datetime(int(split['train_year_end']) + 1, 1, 1)
    train_lower_bound = datetime.datetime(int(split['train_year_start']), 1, 1)

    train = data.loc[(data[year_column] >= train_lower_bound) & (data[year_column] < train_upper_bound)]
    valid = data.loc[(data[year_column] >= train_upper_bound) & (data[year_column] < valid_upper_bound)]
    test = data.loc[(data[year_column] >= valid_upper_bound) & (data[year_column] < test_upper_bound)]

    # Check that disjoint sets.
    assert len(train.index.intersection(valid.index).to_list()) == 0 and\
        len(train.index.intersection(test.index).to_list()) == 0 and\
        len(valid.index.intersection(test.index).to_list()) == 0
    assert data.loc[(data[year_column] >= train_lower_bound) & (data[year_column] < test_upper_bound)].shape[0] ==\
           train.shape[0] + valid.shape[0] + test.shape[0]

    x_train = train[feature_columns].copy()
    y_train = train[label_column].copy()
    x_valid = valid[feature_columns].copy()
    y_valid = valid[label_column].copy()
    x_test = test[feature_columns].copy()
    y_test = test[label_column].copy()
    # Whenever a split is retrieved again check that number of elements still match.
    if 'train_n' in split:
        assert split['train_n'] == x_train.shape[0]
    if 'valid_n' in split:
        assert split['valid_n'] == x_valid.shape[0]
    if 'test_n' in split:
        assert split['test_n'] == x_test.shape[0]

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def derive_column_types(data):
    is_column_numerical = data.apply(lambda s: pd.to_numeric(s.fillna(0), errors='coerce')).notnull().all()
    numerical_columns = data.columns[is_column_numerical].tolist()
    nominal_columns = data.columns[[not i for i in is_column_numerical]].tolist()
    return numerical_columns, nominal_columns


def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
