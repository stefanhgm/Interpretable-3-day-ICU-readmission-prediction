/* Types of items taken from the PDMS. */
CREATE TABLE public.itemtypes (
  	ID int NOT NULL,
  	NAME varchar(127) NOT NULL,
  	PRIMARY KEY (ID)
);

/* Item definitions taken from the PDMS. */
CREATE TABLE public.items (
  	ID int NOT NULL,
  	ITEMTYPE smallint DEFAULT NULL,
  	NAME varchar(127) NOT NULL,
  	ITEMCOMMENT varchar(1024) DEFAULT NULL,
  	ISACTIVE smallint NOT NULL,
  	ISFIXED smallint NOT NULL,
  	ABSOLUTEUPPER real DEFAULT NULL,
  	ABSOLUTELOWER real DEFAULT NULL,
  	QUESTIONABLEUPPER real DEFAULT NULL,
  	QUESTIONABLELOWER real DEFAULT NULL,
  	VALUEPRECISION real DEFAULT NULL,
  	MODIFIEDTIME timestamp(0) DEFAULT NULL,
  	MODIFIEDBYID int DEFAULT NULL,
  	FLUIDCATEGORYID int DEFAULT NULL,
  	CLASSID int DEFAULT NULL,
  	EXPORTTIME int NOT NULL,
  	PRIMARY KEY (ID)
);

/* Values of coded items in PDMS. */
CREATE TABLE public.itemchoice (
	ID serial NOT NULL,
  	ITEMID int NOT NULL,
  	POSITION int NOT NULL,
  	NUMERICVALUE int NOT NULL,
  	SHORTNAME varchar(1024) DEFAULT NULL,
  	LONGNAME varchar(1024) DEFAULT NULL,
  	ISNORMAL int NOT NULL,
  	EXPORTTIME int NOT NULL,
  	PRIMARY KEY (ID)
);

/* Recordings for items taken from the PDMS. Depending on the item type a recording contains different values, so
   we chose a generic design that stores all values as text. */
CREATE TABLE public.recordings (
  	id serial NOT NULL,
	patientid varchar (120) NOT NULL,
	itemid int NOT NULL,
	displaytime timestamp(0) NOT NULL,
	numericvalue_text text,
	numericvalue text,
	textvalue_text text,
	textvalue text,
  	PRIMARY KEY (ID)
);

CREATE INDEX patientid_idx ON public.recordings (patientid);
CREATE INDEX itemid_idx ON public.recordings (itemid);
CREATE INDEX displaytime_idx ON public.recordings (displaytime);
CREATE INDEX numericvalue_text_idx ON public.recordings (numericvalue_text);
CREATE INDEX numericvalue_idx ON public.recordings (numericvalue);
CREATE INDEX textvalue_text_idx ON public.recordings (textvalue_text);
CREATE INDEX textvalue_idx ON public.recordings (textvalue);

/* ICU transfers */
CREATE TABLE public.transfers (
  	id int NOT NULL,
	patientid varchar (120) NOT NULL,
	fallnummer varchar (120) NOT NULL,
	intvon timestamp(0) NOT NULL,
	intbis timestamp(0) NOT NULL,
	station varchar (120) NOT NULL,
	speciality varchar (120) NOT NULL,
	ukmvon timestamp(0) NOT NULL,
	ukmbis timestamp(0),
	discharge varchar (10),
  	PRIMARY KEY (ID)
);

/* Types of variables generated from items. */
CREATE TABLE public.variables (
  	id serial NOT NULL,
  	name varchar(1024) NOT NULL,
  	type varchar(1024) NOT NULL,
  	generatedtime timestamp(0) NOT NULL,
  	PRIMARY KEY (id)
);

/* Values of variables derived from the raw recordings. */
CREATE TABLE public.values (
  	id serial NOT NULL,
  	patientid varchar (120) NOT NULL,
	variablename varchar(1024) NOT NULL,
	displaytime timestamp(0) NOT NULL,
  	numericvalue real default NULL,
  	textvalue text default NULL,
  	PRIMARY KEY (id)
);

/* Remove them for fast insertion performance*/
CREATE INDEX values_patientid_idx ON public.values (patientid);
CREATE INDEX values_variable_idx ON public.values (variablename);
CREATE INDEX values_displaytime_idx ON public.values (displaytime);
CREATE INDEX values_numericvalue_idx ON public.values (numericvalue);
CREATE INDEX values_textvalue_idx ON public.values (textvalue);

/* ICU stays */
CREATE TABLE public.stays (
  	id int NOT NULL,
	patientid varchar (120) NOT NULL,
	fallnummer varchar (120) NOT NULL,
	intvon timestamp(0) NOT NULL,
	intbis timestamp(0) NOT NULL,
	station varchar (1024) NOT NULL,
	speciality varchar (1024) NOT NULL,
	ukmvon timestamp(0) NOT NULL,
	ukmbis timestamp(0),
	discharge varchar (10),
	next_trans_id varchar (1024),
	next_trans_fallnummer varchar (1024),
	next_trans_intvon timestamp(0),
	next_trans_station varchar (1024),
	next_trans_speciality varchar (1024),
	last_per_patient boolean not null,
	last_per_case boolean not null,
	next_trans_consecutive boolean not null,
	next_trans_close boolean not null,
	merged_ids varchar (1024),
	label boolean default NULL,
  	PRIMARY KEY (id)
);

/* Features generated from variables and values. */
CREATE TABLE public.features (
  	id serial NOT NULL,
  	name varchar(1024) NOT NULL,
  	generatedtime timestamp(0) NOT null,
  	datacolumn varchar(1024) NOT NULL,
  	PRIMARY KEY (id)
);

/* Values of features derived from variable values. */
CREATE TABLE public.inputs (
  	id serial NOT NULL,
  	stayid varchar (120) NOT NULL,
	featurename varchar(1024) NOT NULL,
  	numericvalue real default NULL,
  	textvalue text default NULL,
  	PRIMARY KEY (id)
);

/* MIMIC-IV tables */

/* MIMIC ICU stays based on mimic_derived.icustay_detail. */
CREATE TABLE public.mimic_stays (
	id serial NOT NULL,
    subject_id integer,
    hadm_id integer,
    stay_id integer,
    gender character(1),
    dod date,
    admittime timestamp without time zone,
    dischtime timestamp without time zone,
    los_hospital numeric,
    admission_age numeric,
    ethnicity character varying(80),
    hospital_expire_flag smallint,
    hospstay_seq bigint,
    first_hosp_stay boolean,
    icu_intime timestamp without time zone,
    icu_outtime timestamp without time zone,
    los_icu numeric,
    icustay_seq bigint,
    first_icu_stay boolean,
    first_careunit character varying(80),
    last_careunit character varying(80),
    deathtime timestamp without time zone,
    label boolean default null,
    PRIMARY KEY (id)
);

CREATE TABLE public.mimic_features (
  	id serial NOT NULL,
  	name varchar(1024) NOT NULL,
  	generatedtime timestamp(0) NOT null,
  	datacolumn varchar(1024) NOT NULL,
  	PRIMARY KEY (id)
);

CREATE TABLE public.mimic_inputs (
  	id serial NOT NULL,
  	hadm_id varchar (120) NOT NULL,
	featurename varchar(1024) NOT NULL,
  	numericvalue real default NULL,
  	textvalue text default NULL,
  	PRIMARY KEY (id)
);