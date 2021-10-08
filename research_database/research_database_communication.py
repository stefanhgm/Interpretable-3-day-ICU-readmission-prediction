"""
Handling connection to the research database containing data from PDMS.
"""
import datetime as dt
import functools

import psycopg2
import sqlalchemy
from psycopg2 import sql

import pandas as pd


def cur_conn_wrapper(func):
    @functools.wraps(func)
    def decorated(*args, **kwargs):
        args[0].init_cur_conn()
        result = func(*args, **kwargs)
        args[0].close_cur_conn()
        return result
    return decorated


def eng_wrapper(func):
    @functools.wraps(func)
    def decorated(*args, **kwargs):
        args[0].init_eng()
        result = func(*args, **kwargs)
        args[0].close_eng()
        return result
    return decorated


def resolve_data_column_name(var_type, feat_function_name=None):
    if feat_function_name is not None:
        if feat_function_name == 'unique':
            return 'numericvalue'

    if var_type == 'categorical':
        return 'textvalue'
    elif var_type == 'continuous':
        return 'numericvalue'
    else:
        raise Exception("Unexpected variable type.")


class DBConnection:

    def __init__(self, dbname, password):
        self.host = 'localhost'
        self.dbname = dbname
        self.schema = 'public'
        self.user = 'postgres'
        self.port = '5432'
        self.password = password
        # Careful: these are usually not set and the decorators are responsible to initialize them when necessary.
        self.conn = None
        self.cur = None
        self.eng = None

    def init_cur_conn(self):
        self.conn = psycopg2.connect(host=self.host, dbname=self.dbname, user=self.user, port=self.port,
                                     password=self.password, options=f'-c search_path={self.schema}')
        self.cur = self.conn.cursor()

    def close_cur_conn(self):
        self.conn.close()
        self.cur.close()

    def init_eng(self):
        self.eng = sqlalchemy.create_engine('postgresql+psycopg2://' + self.user + ':' + self.password + '@' +
                                            self.host + ':' + self.port + '/' + self.dbname,
                                            connect_args={'options': '-csearch_path={}'.format(self.schema)})

    def close_eng(self):
        self.eng.dispose()

    def get_cur_conn(self):
        self.init_cur_conn()
        return self.cur, self.conn

    @cur_conn_wrapper
    def read_table_from_db(self, schema, table):
        query = sql.SQL("select * from {}.{}").format(sql.Identifier(schema), sql.Identifier(table))
        table_data = pd.read_sql_query(query, self.conn)
        print(f"\tRead {schema}.{table} with shape {table_data.shape}.")
        return table_data

    @cur_conn_wrapper
    @eng_wrapper
    def write_table_to_db(self, schema, table, data):
        query = sql.SQL("delete from {}.{}").format(sql.Identifier(schema), sql.Identifier(table))
        self.cur.execute(query)
        self.conn.commit()
        data.to_sql(table, self.eng, if_exists='append', index=False, method='multi', chunksize=10000)


class ResearchDBConnection(DBConnection):

    def __init__(self, password):
        super().__init__('data', password)

    def read_table_from_db(self, table):
        return super().read_table_from_db(schema='public', table=table)

    def write_table_to_db(self, table, data):
        return super().write_table_to_db(schema='public', table=table, data=data)

    @cur_conn_wrapper
    def read_records_from_db(self, source_columns, item_id=None, pid=None, start=None, end=None):
        assert (start is None and end is None) or (start is not None and end is not None)
        query = sql.SQL("select patientid, itemid, displaytime, {fields} from recordings "
                        + (" where " if item_id is not None or pid is not None or start is not None else " ")
                        + (" itemid = %s " if item_id is not None else " ")
                        + (" and " if item_id is not None and pid is not None else " ")
                        + (" patientid = %s " if pid is not None else " ")
                        + (" and " if (item_id is not None or pid is not None) and start is not None else " ")
                        + (" displaytime >= %s and displaytime <= %s" if start is not None else " ")).format(
            fields=sql.SQL(',').join([sql.Identifier(identifier) for identifier in source_columns]))
        parameters = []
        if item_id is not None:
            parameters.append(str(item_id))
        if pid is not None:
            parameters.append(str(pid))
        if start is not None:
            parameters += [start.strftime('%Y-%m-%d %H:%M:%S'), end.strftime('%Y-%m-%d %H:%M:%S')]
        records = pd.read_sql_query(query, self.conn, params=tuple(parameters))
        # print(f"\tRead records with shape {records.shape}.")
        return records

    @cur_conn_wrapper
    def read_records_per_item(self):
        query = sql.SQL("select itemid, itemtype, name, count from "
                        "(select itemid, count(*) from recordings group by itemid) as r "
                        "left join items on r.itemid = id order by itemid;")
        records_per_item = pd.read_sql_query(query, self.conn)
        return records_per_item

    @cur_conn_wrapper
    @eng_wrapper
    def write_variable_and_values_to_db(self, var_name, var_type, records):
        # Store resulting variables and values in database.
        # Delete existing variable and value entries.
        self.cur.execute(sql.SQL("delete from variables where name = %s"), (var_name,))
        self.cur.execute(sql.SQL("delete from values where variablename = %s"), (var_name,))
        self.conn.commit()
        # Store new variable entry.
        var_generated_time = dt.datetime.now()
        self.cur.execute(sql.SQL("insert into variables (name, type, generatedtime) values (%s, %s, %s)"),
                                (var_name, var_type, var_generated_time,))
        self.conn.commit()
        # Store variable values.
        del records['itemid']
        records['variablename'] = var_name
        records.to_sql('values', self.eng, if_exists='append', index=False, method='multi', chunksize=10000)

    @eng_wrapper
    def write_values_to_db(self, var_name, records):
        # Store variable values.
        records['variablename'] = var_name
        records.to_sql('values', self.eng, if_exists='append', index=False, method='multi', chunksize=10000)

    @cur_conn_wrapper
    def read_transfers_with_min_max_variable_from_db(self, var_name):
        query = sql.SQL("SELECT id, cases_selection.patientid, fallnummer, intvon, intbis, station, speciality, "
                        "ukmvon, ukmbis, discharge, min(displaytime), max(displaytime) "
                        "FROM (SELECT * FROM transfers) AS cases_selection "
                        "JOIN (SELECT patientid, displaytime FROM values WHERE values.variablename = %s) AS hf "
                        "ON cases_selection.fallnummer = hf.patientid "
                        "where hf.displaytime >= cases_selection.intvon and hf.displaytime <= cases_selection.intbis "
                        "group by id, cases_selection.patientid, fallnummer, intvon, intbis, station, speciality, "
                        "ukmvon, ukmbis, discharge;")
        transfers = pd.read_sql_query(query, self.conn, params=(var_name,))
        return transfers

    @cur_conn_wrapper
    @eng_wrapper
    def write_stays_to_db(self, stays):
        self.cur.execute(sql.SQL("delete from stays"))
        self.conn.commit()
        string_columns = ['patientid', 'fallnummer', 'station', 'speciality', 'discharge',
                          'next_trans_fallnummer', 'next_trans_station', 'next_trans_speciality', 'merged_ids']
        stays[string_columns] = stays[string_columns].astype(str)
        stays['next_trans_id'] = stays['next_trans_id'].fillna(-1)
        stays['next_trans_id'] = stays['next_trans_id'].astype('int')
        stays.to_sql('stays', self.eng, if_exists='append', index=False, method='multi', chunksize=10000)

    @cur_conn_wrapper
    def get_stays_without_value_per_hosp_stay(self, var_name):
        query = sql.SQL("select * from stays left join "
                        "(select * from values where values.variablename = %s) as values "
                        "on stays.fallnummer = values.patientid and values.displaytime >= stays.ukmvon and "
                        "values.displaytime <= stays.intbis where "
                        "(values.numericvalue is null and values.textvalue is null);")
        stays = pd.read_sql_query(query, self.conn, params=(var_name,))
        return stays

    @cur_conn_wrapper
    def get_stays_without_value_per_icu_stay(self, var_name):
        query = sql.SQL("select * from stays left join "
                        "(select * from values where values.variablename = %s) as values "
                        "on stays.fallnummer = values.patientid and values.displaytime >= stays.intvon and "
                        "values.displaytime <= stays.intbis where "
                        "(values.numericvalue is null and values.textvalue is null);")
        stays = pd.read_sql_query(query, self.conn, params=(var_name,))
        return stays

    @cur_conn_wrapper
    @eng_wrapper
    def write_mimic_stays_to_db(self, stays):
        self.cur.execute(sql.SQL("delete from mimic_stays"))
        self.conn.commit()
        string_columns = ['eventtype', 'careunit', 'admission_type', 'discharge_location', 'next_trans_careunit',
                          'merged_ids']
        stays[string_columns] = stays[string_columns].astype(str)
        stays['subject_id'] = stays['subject_id'].astype('int')
        stays['hadm_id'] = stays['hadm_id'].fillna(-1)
        stays['hadm_id'] = stays['hadm_id'].astype('int')
        stays['transfer_id'] = stays['transfer_id'].astype('int')
        stays['next_trans_transfer_id'] = stays['next_trans_transfer_id'].fillna(-1)
        stays['next_trans_transfer_id'] = stays['next_trans_transfer_id'].astype('int')
        stays['next_trans_hadm_id'] = stays['next_trans_hadm_id'].fillna(-1)
        stays['next_trans_hadm_id'] = stays['next_trans_hadm_id'].astype('int')
        stays.to_sql('mimic_stays', self.eng, if_exists='append', index=False, method='multi', chunksize=10000)

    @cur_conn_wrapper
    def read_values_from_db(self, var_name, var_type, case_id=None):
        data_column = resolve_data_column_name(var_type)
        if case_id is not None:
            if type(case_id) == list:
                query = sql.SQL("select patientid, displaytime, {field} from values "
                                "where variablename = %s and patientid in %s").format(field=sql.SQL(data_column))
                case_id = tuple(case_id)
            else:
                query = sql.SQL("select patientid, displaytime, {field} from values "
                                "where variablename = %s and patientid = %s")\
                    .format(field=sql.SQL(data_column))
            values = pd.read_sql_query(query, self.conn, params=(var_name, case_id))
        else:
            query = sql.SQL("select patientid, displaytime, {field} from values where variablename = %s")\
                .format(field=sql.SQL(data_column))
            values = pd.read_sql_query(query, self.conn, params=(var_name,))
        values = values.rename(columns={values.columns[2]: 'data'})
        return values

    @cur_conn_wrapper
    @eng_wrapper
    def write_features_and_inputs_to_db(self, feat_names, inputs):
        # Store resulting feature and inputs in database.
        # Delete existing feature and feature entries.
        if type(feat_names) is not list:
            feat_names = [feat_names]
        for feat_name in feat_names:
            self.cur.execute(sql.SQL("delete from features where name = %s"), (feat_name,))
            self.cur.execute(sql.SQL("delete from inputs where featurename = %s"), (feat_name,))
        self.conn.commit()
        # Store new feature entry.
        feat_generated_time = dt.datetime.now()
        for feat_name in feat_names:
            self.cur.execute(sql.SQL("insert into features (name, generatedtime, datacolumn) values (%s, %s, %s)"),
                                    (feat_name, feat_generated_time, inputs.columns.tolist()[-1]))
        self.conn.commit()
        # Store variable values.
        inputs.to_sql('inputs', self.eng, if_exists='append', index=False, method='multi', chunksize=10000)

    @cur_conn_wrapper
    def read_inputs_from_db(self):
        query = sql.SQL("select stayid, featurename, numericvalue, textvalue from inputs")
        inputs = pd.read_sql_query(query, self.conn)
        return inputs

    @cur_conn_wrapper
    def read_inputs_for_stay_from_db(self, stay_id):
        query = sql.SQL("select featurename, numericvalue, textvalue from inputs where stayid = %s")
        inputs = pd.read_sql_query(query, self.conn, params=(str(stay_id),))
        return inputs


class MIMICDBConnection(DBConnection):

    def __init__(self, password):
        super().__init__('mimic-iv', password)

    def read_table_from_db(self, table):
        return super().read_table_from_db(schema='mimic_core', table=table)

    def read_table_from_db(self, schema, table):
        return super().read_table_from_db(schema=schema, table=table)

    def write_table_to_db(self, table, data):
        return super().write_table_to_db(schema='mimic_core', table=table, data=data)

    @cur_conn_wrapper
    def read_adults_icu_transfers_from_transfers_and_admissions(self, icus):
        query = sql.SQL("select trans.subject_id, trans.hadm_id, trans.transfer_id, trans.eventtype, trans.careunit,"
                        " trans.intime, trans.outtime, adm.admission_type, adm.admittime, adm.dischtime, adm.deathtime,"
                        " adm.discharge_location from mimic_core.transfers trans"
                        " inner join mimic_core.patients patients on trans.subject_id = patients.subject_id"
                        " inner join mimic_core.admissions adm on trans.hadm_id = adm.hadm_id"
                        " where patients.anchor_age >= 18 and trans.careunit in %s")
        transfers = pd.read_sql_query(query, self.conn, params=(tuple(icus),))
        return transfers

    @cur_conn_wrapper
    def read_values_from_db(self, item_id, table, case_id):
        time_field = 'charttime'
        if table == 'chartevents':
            data_field = 'valuenum'
        elif table == 'datetimeevents':
            data_field = 'value'
        elif table == 'inputevents':
            data_field = 'ordercategoryname'
            time_field = 'starttime'
        elif table == 'outputevents':
            data_field = 'value'
        elif table == 'procedureevents':
            data_field = 'value'
            time_field = 'endtime'
        else:
            raise ValueError("Bad table name")

        query = sql.SQL("select subject_id, {time_field}, {data_field} from mimic_icu.{table} "
                        "where itemid = %s and subject_id in %s")\
            .format(time_field=sql.SQL(time_field), data_field=sql.SQL(data_field), table=sql.SQL(table))
        case_id = tuple(case_id)
        values = pd.read_sql_query(query, self.conn, params=(item_id, case_id))
        values = values.rename(columns={values.columns[1]: 'charttime'})
        values = values.rename(columns={values.columns[2]: 'data'})
        return values

    @cur_conn_wrapper
    def read_icu_stays_with_min_max_hf_from_db(self):
        query = sql.SQL("SELECT  subject_id , hadm_id , stay_id , intime, outtime, min(charttime), max(charttime) "
                        "FROM (SELECT subject_id, hadm_id, stay_id, intime, outtime "
                        "from \"mimic-iv\".mimic_icu.icustays i) AS cases_selection "
                        "JOIN (SELECT subject_id as s_id, charttime FROM \"mimic-iv\".mimic_icu.chartevents c "
                        "WHERE c.itemid = 220045) AS hf ON cases_selection.subject_id = hf.s_id "
                        " where hf.charttime >= cases_selection.intime and hf.charttime <= cases_selection.outtime "
                        "group by subject_id , hadm_id , stay_id , intime, outtime")
        icu_stays = pd.read_sql_query(query, self.conn)
        return icu_stays
