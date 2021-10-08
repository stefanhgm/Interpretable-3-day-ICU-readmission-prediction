import argparse
import collections
import itertools
import sys
import time
from datetime import datetime

import os

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from interpret.glassbox.ebm.utils import EBMUtils
from interpret.utils import unify_data
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from experiments.variables import seed
from helper.io import read_item_processing_descriptions_from_excel
from helper.util import read_features_years_labels, create_splits, get_split, output_wrapper
from experiments.custom_ebm.ebm_enable_unk_min_samples_bin import ExplainableBoostingClassifier

import logging

# LR parameters.
lr_params = {
    'penalty': ['l1', 'l2'],
    # Took Cs from this paper + additional small values: https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf.
    # First experiments showed that l1 and l2 over fit clearly for C > 1, so reduced Cs to 2^3.
    'C': [2 ** x for x in range(-20, 4)],
    # Also try some larger tolerances for models with proper generalization.
    'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001],
    # Use this common solver that supports l1/l2 norms.
    'solver': ['liblinear'],
    'max_iter': [10, 100, 500],
    'random_state': [seed]
}

# Chose best LR with pr valid of 0.07959 trained on all 1423 features.
# Chose 100 over 500 iter bc only 11 used during training.
best_lr_params_full_features = {
    'penalty': 'l1',
    'C': 0.125,
    'tol': 0.01,
    'solver': 'liblinear',
    'max_iter': 100,
    'random_state': seed
}

most_important_130_feat =\
    ['Blood volume out (extrapolate 7d) [mL]_nan', 'paO2/FiO2 (max 1d) [mmHg/FiO2]', 'Age (static all data) [years]',
     'Beta blocking agents (unique 1d)', 'LDH (trend per day 1d) [U/L]', 'pH (max 1d)',
     'Plasma expander volume in (extrapolate 7d) [mL]_nan', 'Chloride (min 3d) [mmol/L]', 'BE (median 3d) [mmol/L]',
     'pCO2 (median 12h) [mmHg]', 'Beta blocking agents (unique 3d)', 'paO2/FiO2 (iqr 1d) [mmHg/FiO2]',
     'FiO2 (median 12h) [%]', 'Vasodilatators used in cardiac diseases (unique 1d)', 'CK-MB (median 1d) [U/L]',
     'Gender (static all data)_Male', 'Heart rate (max 12h) [bpm]', 'Antibacterials for systemic use (unique 7d)',
     'Estimated respiratory rate (iqr 4h)', 'Leucocytes (min 1d) [thousand/μL]', 'Phosphate (min 7d) [mg/dL]',
     'Chloride (max 3d) [mmol/L]', 'pH (iqr 1d)', 'O2Hb (min 12h) [%]', 'LDH (median 1d) [U/L]',
     'Leucocytes (max 7d) [thousand/μL]', 'Systolic blood pressure (iqr 1d) [mmHg]',
     'Tubus exists (has-true per icu stay)_1.0', 'T4 free (max 7d) [ng/dL]', 'pH (trend per day 3d)',
     'Heart rate (iqr 12h) [bpm]', 'Opioids (unique 7d)', 'Blood Urea Nitrogen (median 3d) [mg/dL]',
     'Body core temperature (median 4h) [°C]', 'BE (iqr 1d) [mmol/L]', 'Alkaline phosphatase (median 7d) [U/L]',
     'MetHb (min 3d) [%]', 'Chloride (max 12h) [mmol/L]', 'pCO2 (median 3d) [mmHg]', 'Glucose (median 3d) [mg/dL]',
     'Heart rate (median 12h) [bpm]', 'T3 free (max 7d) [ng/dL]', 'pCO2 (max 1d) [mmHg]', 'PTT (median 1d) [s]',
     'Diastolic blood pressure (median 12h) [mmHg]', 'Lactate (min 3d) [mmol/L]',
     'Responsible clinic (static per hospital stay)_Orthopedic surgery', 'Potassium (median 12h) [mmol/L]',
     'Tubus exists (days since last application per icu stay)_nan',
     'Drugs for peptic ulcer and reflux without PPIs (unique 3d)', 'Body core temperature (max 12h) [°C]',
     'ICU length of stay [days]', 'pO2 (max 1d) [mmHg]', 'RAS scale (median 3d)', 'RAS scale (max 3d)',
     'MCV (iqr 7d) [fL]', 'Vasodilatators used in cardiac diseases (unique 7d)', 'Thrombocytes (max 3d) [thousand/μL]',
     'Ultrafiltrate volume out (extrapolate 1d) [mL]_nan', 'pH (iqr 3d)', 'Anion gap (iqr 12h) [mEq/L]',
     'Alkaline phosphatase (min 3d) [U/L]', 'Admission origin (static per hospital stay)_operating room',
     'RAS scale (min 12h)', 'Heart rate (iqr 4h) [bpm]', 'Systolic blood pressure (min 1d) [mmHg]',
     'Sodium (median 1d) [mmol/L]', 'CK (min 1d) [U/L]', 'RAS scale (min 3d)', 'Weight (static all data) [kg]',
     'Urine volume out (extrapolate 3d) [mL]', 'Drugs used in diabetes (unique 7d)_nan',
     'Antibacterials for systemic use (unique 1d)_nan', 'Lactate (min 12h) [mmol/L]', 'PTT (min 1d) [s]',
     'BMI (static all data) [kg/m2]', 'Antibacterials for systemic use (unique 1d)',
     'C-reactive protein (iqr 7d) [mg/dL]', 'Drugs for constipation (has entry 1d)', 'Gamma-GT (iqr 7d) [U/L]',
     'BE (min 12h) [mmol/L]', 'Fibrinogene (max 1d) [mg/dL]', 'Bicarbonate (max 12h) [mmol/L]',
     'Estimated respiratory rate (min 12h)', 'Hematocrit (min 3d) [%]', 'COHb (iqr 12h) [%]',
     'Thrombocytes (trend per day 7d) [thousand/μL]', 'Antiarrhythmics, class I and III (unique 1d)',
     'Body core temperature (min 4h) [°C]', 'Type of admission (static per hospital stay)_Scheduled surgical',
     'GCS Verbal (trend per day 7d)', 'Blood volume out (extrapolate 3d) [mL]', 'Lactate (trend per day 3d) [mmol/L]',
     'Blood volume in (extrapolate 3d) [mL]', 'Glucose (min 1d) [mg/dL]', 'Sodium (max 1d) [mmol/L]',
     'Antithrombotic agents prophylactic dosage (has-true per icu stay)_nan', 'O2 saturation (trend per day 1d) [%]',
     'Drugs for constipation (has entry 1d)_nan', 'Prothrombin time (INR) (median 3d)',
     'C-reactive protein (median 1d) [mg/dL]', 'Glucose (trend per day 12h) [mg/dL]', 'Heart rate (max 1d) [bpm]',
     'RHb (iqr 12h)', 'GCS Eye (trend per day 3d)', 'C-reactive protein (median 7d) [mg/dL]',
     'Body core temperature (min 1d) [°C]', 'GCS Verbal (max 3d)', 'Bicarbonate (iqr 3d) [mmol/L]',
     'Antithrombotic agents excl. Platelet inhibitors and enzymes (has entry 7d)_nan',
     'Antithrombotic agents excl. Platelet inhibitors and enzymes (unique 1d)', 'Non-opioid analgetics (unique 7d)',
     'Procalcitonin (max 1d) [ng/mL]_nan', 'Gender (static all data)_Female', 'Urine volume out (extrapolate 7d) [mL]',
     'Antithrombotic agents therapeutic dosage (has-true per icu stay)_nan', 'PAP - PEEP (iqr 4h) [mmHg]',
     'PTT (iqr 1d) [s]', 'Heart rate (median 4h) [bpm]',
     'Responsible clinic (static per hospital stay)_Thoracic surgery', 'TSH (iqr 7d) [ μU/mL]_nan',
     'Phosphate (median 1d) [mg/dL]_nan', 'Glucose (iqr 12h) [mg/dL]', 'GCS Verbal (min 7d)', 'MCV (min 7d) [fL]',
     'Vasodilatators used in cardiac diseases (unique 3d)', 'Urine volume out (extrapolate 1d) [mL]',
     'eGFR (trend per day 7d) [L]', 'Glucose (max 3d) [mg/dL]', 'pH (trend per day 12h)']


# Best LR with pr valid of 0.09531 trained on 130 features.
# Same results for max_iter 10, 100, 500, choose 100 as tradeoff.
best_lr_params_130_features = {
    'penalty': 'l1',
    'C': 1,
    'tol': 0.1,
    'solver': 'liblinear',
    'max_iter': 100,
    'random_state': seed
}

best_lr_features = \
    ['Admission origin (static per hospital stay)_operating room', 'PTT (median 1d) [s]',
     'Drugs for constipation (has entry 1d)_nan', 'Age (static all data) [years]',
     'Selective calcium channel blockers with mainly vascular effects (has entry 3d)',
     'paO2/FiO2 (min 12h) [mmHg/FiO2]', 'Drugs used in diabetes (has entry 7d)', 'Blood Urea Nitrogen (min 3d) [mg/dL]',
     'Heart rate (min 1d) [bpm]', 'Drugs for constipation (unique 1d)_nan', 'RAS scale (max 3d)',
     'BE (iqr 3d) [mmol/L]', 'Blood volume out (extrapolate 7d) [mL]_nan',
     'Chest tube exists (has-true per icu stay)_1.0', 'Drugs for constipation (has entry 1d)',
     'Drugs used in diabetes (has entry 7d)_nan', 'RAS scale (median 3d)',
     'Drugs for obstructive airway diseases (has entry 7d)', 'Stool volume out (extrapolate 3d) [mL]',
     'Chest tube exists (days since last application per icu stay)_nan',
     'Selective calcium channel blockers with mainly vascular effects (has entry 3d)_nan',
     'Drugs for obstructive airway diseases (has entry 7d)_nan',
     'Anxiolytics, hypnotics and sedatives (has entry 7d)_nan',
     'Type of admission (static per hospital stay)_Scheduled surgical',
     'Selective calcium channel blockers with mainly vascular effects (unique 3d)_nan',
     'ACE inhibitors plain and combinations (has entry 7d)', 'Chest tube exists (has-true per icu stay)_nan',
     'Bicarbonate (iqr 3d) [mmol/L]', 'Antithrombotic agents prophylactic dosage (has-true per icu stay)_1.0',
     'Responsible clinic (static per hospital stay)_Cardiothoracic surgery', 'BMI (static all data) [kg/m2]',
     'O2Hb (median 12h) [%]', 'Non-opioid analgetics (has entry 1d)',
     'Anxiolytics, hypnotics and sedatives (unique 7d)_nan', 'C-reactive protein (max 3d) [mg/dL]',
     'FiO2 (min 12h) [%]', 'ACE inhibitors plain and combinations (has entry 7d)_nan', 'PAP - PEEP (min 1d) [mmHg]',
     'PAP - PEEP (trend per day 12h) [mmHg]', 'Antihypertensiva (unique 1d)', 'Antihypertensiva (unique 3d)',
     'Plasma expander volume in (extrapolate 7d) [mL]_nan', 'PAP - PEEP (median 1d) [mmHg]',
     'Tubus exists (days since last application per icu stay)_nan', 'PAP - PEEP (iqr 12h) [mmHg]_nan',
     'Non-opioid analgetics (has entry 1d)_nan', 'Feeding system exists (days since last application per icu stay)_nan',
     'Antithrombotic agents prophylactic dosage (days since last application per icu stay)_nan',
     'ACE inhibitors plain and combinations (has entry 1d)', 'pH (median 1d)', 'Thrombocytes (max 3d) [thousand/μL]',
     'Thrombocytes (max 1d) [thousand/μL]', 'Thrombocytes (max 7d) [thousand/μL]',
     'Thrombocytes (median 3d) [thousand/μL]', 'ACE inhibitors plain and combinations (unique 3d)',
     'ACE inhibitors plain and combinations (unique 7d)_nan', 'PAP - PEEP (iqr 1d) [mmHg]_nan', 'FiO2 (max 12h) [%]',
     'C-reactive protein (max 7d) [mg/dL]', 'Antithrombotic agents prophylactic dosage (has-true per icu stay)_nan',
     'Thrombocytes (median 1d) [thousand/μL]', 'Anxiolytics, hypnotics and sedatives (has entry 7d)',
     'Anion gap (trend per day 1d) [mEq/L]', 'Anxiolytics, hypnotics and sedatives (has entry 3d)_nan',
     'Drugs for constipation (has entry 3d)', 'Feeding system exists (has-true per icu stay)_nan',
     'Procalcitonin (iqr 7d) [ng/mL]_nan', 'Procalcitonin (iqr 3d) [ng/mL]_nan', 'Procalcitonin (max 7d) [ng/mL]_nan',
     'Procalcitonin (max 3d) [ng/mL]_nan', 'Procalcitonin (iqr 1d) [ng/mL]', 'O2Hb (max 12h) [%]',
     'Procalcitonin (median 3d) [ng/mL]_nan', 'Phosphate (iqr 1d) [mg/dL]_nan', 'Phosphate (median 7d) [mg/dL]',
     'Phosphate (trend per day 1d) [mg/dL]', 'Phosphate (max 1d) [mg/dL]_nan', 'Procalcitonin (median 7d) [ng/mL]_nan',
     'Phosphate (median 1d) [mg/dL]_nan', 'Procalcitonin (min 3d) [ng/mL]_nan', 'Phosphate (min 1d) [mg/dL]_nan',
     'Mean blood pressure (min 1d) [mmHg]', 'Procalcitonin (trend per day 3d) [ng/mL]_nan',
     'paO2/FiO2 (max 1d) [mmHg/FiO2]', 'paO2/FiO2 (iqr 1d) [mmHg/FiO2]', 'Phosphate (trend per day 1d) [mg/dL]_nan',
     'Mean blood pressure (min 12h) [mmHg]',
     'Selective calcium channel blockers with mainly vascular effects (unique 3d)',
     'Non-opioid analgetics (unique 1d)_nan', 'Phosphate (max 1d) [mg/dL]',
     'ACE inhibitors plain and combinations (has entry 1d)_nan', 'pO2 (iqr 1d) [mmHg]', 'pH (iqr 1d)', 'pH (max 1d)',
     'pH (median 3d)', 'Tubus exists (has-true per icu stay)_nan', 'Phosphate (min 7d) [mg/dL]',
     'Erythrocytes (max 3d) [millions/μL]', 'Erythrocytes (iqr 7d) [millions/μL]',
     'Erythrocytes (min 3d) [millions/μL]', 'Erythrocytes (iqr 1d) [millions/μL]',
     'Erythrocytes (max 7d) [millions/μL]', 'BE (median 3d) [mmol/L]', 'Bicarbonate (max 12h) [mmol/L]', 'pH (max 12h)',
     'Bicarbonate (median 3d) [mmol/L]', 'Bicarbonate (median 12h) [mmol/L]', 'Bicarbonate (max 1d) [mmol/L]',
     'Anesthestics (unique 3d)', 'Is on automatic ventilation (days since last application per icu stay)_nan',
     'Blood Urea Nitrogen (min 7d) [mg/dL]',
     'Tracheal secretion cleaned (days since last application per icu stay)_nan', 'Anesthestics (has entry 7d)_nan',
     'Is on automatic ventilation (has-true per icu stay)_nan', 'Drugs for constipation (has entry 3d)_nan',
     'Drugs for constipation (has entry 7d)', 'pH (median 12h)', 'BE (max 3d) [mmol/L]', 'pCO2 (median 12h) [mmHg]',
     'pCO2 (median 1d) [mmHg]', 'pO2 (max 1d) [mmHg]', 'Procalcitonin (min 7d) [ng/mL]_nan',
     'paO2/FiO2 (median 1d) [mmHg/FiO2]', 'Erythrocytes (median 3d) [millions/μL]', 'pO2 (median 3d) [mmHg]',
     'Procalcitonin (iqr 1d) [ng/mL]_nan', 'Drugs for constipation (unique 3d)_nan', 'Anesthestics (has entry 3d)',
     'Drugs for constipation (unique 7d)', 'Drugs for constipation (unique 3d)']


# EBM parameters.
# Changed 'max_leaves' from [2, 3, 4, 5, 6], because 6 already best in first experiments.
# Kept outer_bags=8 bc otherwise results too noisy.
ebm_params = {
    'learning_rate': [10 ** x for x in range(-12, 2)],
    'min_samples_leaf': [2],  # Not necessary to tune, bc mostly determined by min_samples_bin.
    'max_leaves': [2, 4, 6, 8, 10],
    'interactions': [0],
    # First round of experiments with [5, 25, 50, 100, 200] gave 25 for 80 rf, but risk function fluctuating too much.
    # Only for 200 they were sufficiently smooth, so try values around 200.
    'min_samples_bin': [100, 150, 200, 250, 300, 350, 400],
    'binning': ['quantile'],
    'max_rounds': [5000],
    'random_state': [seed],
    'outer_bags': [8]
}

# Chose best EBM with pr valid of 0.09284 trained on all 1423 features.
best_ebm_params_full_features = {
    'learning_rate': 0.01,
    'min_samples_leaf': 2,
    'max_leaves': 8,
    'interactions': 0,
    'min_samples_bin': 200,
    'binning': 'quantile',
    'max_rounds': 5000,
    'random_state': seed,
    'outer_bags': 8
}

most_important_80_rf = \
    ['PTT (max 1d) [s]', 'PTT (max 3d) [s]', 'PTT (min 7d) [s]', 'PTT (median 1d) [s]', 'PTT (median 7d) [s]',
     'PTT (median 3d) [s]', 'Length of stay before ICU [days]', 'Chloride (trend per day 3d) [mmol/L]',
     'Infusion volume in (extrapolate 7d) [mL]', 'Blood volume out (extrapolate 7d) [mL]',
     'Systolic blood pressure (iqr 12h) [mmHg]', 'PTT (min 3d) [s]', 'PTT (max 7d) [s]', 'Gamma-GT (max 7d) [U/L]',
     'Gamma-GT (max 1d) [U/L]', 'Chloride (trend per day 1d) [mmol/L]', 'Heart rate (min 12h) [bpm]',
     'Heart rate (min 4h) [bpm]', 'O2 saturation (min 4h) [%]', 'PTT (iqr 3d) [s]', 'pH (trend per day 3d)',
     'Chloride (max 1d) [mmol/L]', 'Tubus exists (days since last application per icu stay)', 'BE (min 12h) [mmol/L]',
     'Hemoglobin (trend per day 1d) [mmol/L]', 'Gamma-GT (max 3d) [U/L]', 'Body core temperature (min 1d) [°C]',
     'O2Hb (trend per day 3d) [%]', 'Blood volume out (extrapolate 1d) [mL]', 'FiO2 (max 12h) [%]', 'PTT (min 1d) [s]',
     'Creatinine (median 1d) [mg/dL]', 'BE (median 12h) [mmol/L]', 'Gamma-GT (min 1d) [U/L]',
     'Mean blood pressure (median 12h) [mmHg]', 'Gamma-GT (median 1d) [U/L]', 'PTT (iqr 7d) [s]',
     'Gamma-GT (min 7d) [U/L]', 'CK-MB (median 7d) [U/L]', 'FiO2 (median 12h) [%]', 'CK (median 7d) [U/L]',
     'CK-MB (iqr 3d) [U/L]', 'Sodium (max 3d) [mmol/L]', 'Gamma-GT (median 7d) [U/L]', 'Phosphate (min 3d) [mg/dL]',
     'Is on automatic ventilation (days since last application per icu stay)', 'Age (static all data) [years]',
     'Urine volume out (extrapolate 1d) [mL]', 'O2 saturation (min 12h) [%]', 'Chloride (min 3d) [mmol/L]',
     'Heart rate (median 12h) [bpm]', 'Heart rate (median 4h) [bpm]', 'BE (trend per day 3d) [mmol/L]',
     'Sodium (trend per day 3d) [mmol/L]', 'Creatinine (max 1d) [mg/dL]', 'Calcium (iqr 3d) [mmol/L]',
     'O2Hb (min 3d) [%]', 'Bicarbonate (trend per day 3d) [mmol/L]', 'Hemoglobin (median 12h) [mmol/L]',
     'O2Hb (iqr 3d) [%]', 'Antithrombotic agents prophylactic dosage (days since last application per icu stay)',
     'Infusion volume in (extrapolate 3d) [mL]', 'Blood volume out (extrapolate 3d) [mL]',
     'CK-MB (trend per day 3d) [U/L]', 'pH (max 1d)', 'Phosphate (max 7d) [mg/dL]', 'CK-MB (max 7d) [U/L]',
     'Bicarbonate (min 12h) [mmol/L]', 'Mean blood pressure (trend per day 1d) [mmHg]', 'Hemoglobin (min 3d) [mmol/L]',
     'Lactate (max 3d) [mmol/L]', 'Hematocrit (trend per day 1d) [%]', 'BE (max 12h) [mmol/L]',
     'Glucose (trend per day 3d) [mg/dL]', 'BE (trend per day 1d) [mmol/L]', 'Phosphate (min 1d) [mg/dL]',
     'CK (min 7d) [U/L]', 'Chloride (median 3d) [mmol/L]', 'Mean blood pressure (iqr 4h) [mmHg]',
     'Heart rate (iqr 1d) [bpm]']

# Chose the best EBM with 80 rf for bin_sizes 100, 150, ..., 400.
# Except for 250 and 400 the best parameters the same.
best_ebm_params_80_risk_functions_default = {
    'min_samples_leaf': 2,
    'interactions': 0,
    'binning': 'quantile',
    'max_rounds': 5000,
    'random_state': seed,
    'outer_bags': 8
}
best_ebm_params_80_risk_functions_bin_size_100 = {
    **best_ebm_params_80_risk_functions_default,
    'min_samples_bin': 100,
    'learning_rate': 0.1,
    'max_leaves': 4,
}
best_ebm_params_80_risk_functions_bin_size_150 = {
    **best_ebm_params_80_risk_functions_default,
    'min_samples_bin': 150,
    'learning_rate': 0.1,
    'max_leaves': 4,
}
best_ebm_params_80_risk_functions_bin_size_200 = {
    **best_ebm_params_80_risk_functions_default,
    'min_samples_bin': 200,
    'learning_rate': 0.1,
    'max_leaves': 4,
}
best_ebm_params_80_risk_functions_bin_size_250 = {
    **best_ebm_params_80_risk_functions_default,
    'min_samples_bin': 250,
    'learning_rate': 1,
    'max_leaves': 2,
}
best_ebm_params_80_risk_functions_bin_size_300 = {
    **best_ebm_params_80_risk_functions_default,
    'min_samples_bin': 300,
    'learning_rate': 0.1,
    'max_leaves': 4,
}
best_ebm_params_80_risk_functions_bin_size_350 = {
    **best_ebm_params_80_risk_functions_default,
    'min_samples_bin': 350,
    'learning_rate': 0.1,
    'max_leaves': 4,
}
best_ebm_params_80_risk_functions_bin_size_400 = {
    **best_ebm_params_80_risk_functions_default,
    'min_samples_bin': 400,
    'learning_rate': 1,
    'max_leaves': 2,
}

# Another parameter tuning showed no relevant difference for other parameter settings.
best_ebm_params_1d_model = best_ebm_params_80_risk_functions_bin_size_200
best_ebm_params_2d_model = best_ebm_params_80_risk_functions_bin_size_200

best_1d_ebm_risk_functions =\
    ['CK (median 7d) [U/L]', 'Chloride (trend per day 3d) [mmol/L]', 'PTT (min 3d) [s]', 'pH (median 1d)',
     'Blood volume out (extrapolate 7d) [mL]', 'Hematocrit (max 3d) [%]', 'Age (static all data) [years]',
     'BE (median 12h) [mmol/L]', 'Length of stay before ICU [days]',
     'Antithrombotic agents prophylactic dosage (days since last application per icu stay)',
     'BE (trend per day 3d) [mmol/L]', 'Phosphate (max 1d) [mg/dL]', 'paO2/FiO2 (median 1d) [mmHg/FiO2]',
     'PTT (max 3d) [s]', 'CK-MB (median 3d) [U/L]', 'Potassium (median 1d) [mmol/L]',
     'Is on automatic ventilation (days since last application per icu stay)', 'RAS scale (max 3d)',
     'pO2 (min 12h) [mmHg]', 'O2 saturation (min 12h) [%]', 'Calcium (trend per day 3d) [mmol/L]',
     'Heart rate (min 4h) [bpm]', 'Body core temperature (min 1d) [°C]', 'pCO2 (median 1d) [mmHg]',
     'CK-MB (max 3d) [U/L]', 'Leucocytes (trend per day 3d) [thousand/μL]', 'Blood Urea Nitrogen (min 3d) [mg/dL]',
     'Urine volume out (extrapolate 1d) [mL]', 'pCO2 (iqr 1d) [mmHg]', 'PTT (max 1d) [s]', 'BE (iqr 3d) [mmol/L]',
     'Chloride (min 1d) [mmol/L]', 'paO2/FiO2 (trend per day 3d) [mmHg/FiO2]', 'eGFR (trend per day 7d) [L]',
     'Drugs for constipation (unique 1d)', 'Bilirubin total (max 7d) [mg/dL]', 'pH (median 3d)',
     'Lactate (max 3d) [mmol/L]', 'Sodium (trend per day 3d) [mmol/L]', 'Gamma-GT (median 7d) [U/L]',
     'Hemoglobin (max 3d) [mmol/L]', 'Leucocytes (median 1d) [thousand/μL]', 'PTT (min 7d) [s]',
     'Diastolic blood pressure (median 1d) [mmHg]', 'paO2/FiO2 (median 3d) [mmHg/FiO2]',
     'Estimated respiratory rate (median 1d)', 'Thrombocytes (trend per day 7d) [thousand/μL]',
     'Body core temperature (min 4h) [°C]', 'PTT (max 7d) [s]', 'RAS scale (trend per day 12h)',
     'Leucocytes (iqr 3d) [thousand/μL]', 'pH (trend per day 3d)', 'Urine volume out (extrapolate 7d) [mL]',
     'RAS scale (max 1d)', 'Sodium (median 3d) [mmol/L]', 'Phosphate (min 7d) [mg/dL]',
     'Blood volume out (extrapolate 3d) [mL]', 'Heart rate (min 1d) [bpm]', 'Systolic blood pressure (iqr 12h) [mmHg]',
     'RHb (median 12h)', 'Body core temperature (median 1d) [°C]', 'Glucose (median 3d) [mg/dL]',
     'Body core temperature (trend per day 1d) [°C]', 'pO2 (iqr 12h) [mmHg]', 'Mean blood pressure (median 4h) [mmHg]',
     'CK (min 7d) [U/L]', 'Lactate (min 12h) [mmol/L]', 'BE (min 12h) [mmol/L]', 'BE (iqr 1d) [mmol/L]',
     'pCO2 (min 3d) [mmHg]', 'GCS score (min 3d)', 'pH (iqr 1d)', 'MetHb (min 12h) [%]', 'Hematocrit (median 12h) [%]',
     'Tubus exists (days since last application per icu stay)', 'Mean blood pressure (median 12h) [mmHg]',
     'Procalcitonin (max 7d) [ng/mL]', 'Heart rate (iqr 4h) [bpm]', 'C-reactive protein (max 3d) [mg/dL]',
     'Calcium (max 1d) [mmol/L]']

# Parameters determined by 2D risk function selection ov bin sizes 4, 6, 8, 10, 12, 14 over 20 iterations.
best_2d_ebm_risk_functions = [(6, 30), (5, 56), (34, 41), (4, 76), (41, 56)]
best_2d_bin_size = 4

# Risk functions removed after inspection through team of doctors.
removed_risk_functions = [
    'PTT (max 1d) [s]',
    'PTT (max 3d) [s]',
    'Calcium (trend per day 3d) [mmol/L]',
    'Body core temperature (min 1d) [°C]',
    'RAS scale (trend per day 12h)',
    'BE (iqr 3d) [mmol/L]',
    'PTT (min 7d) [s]',
    'PTT (min 3d) [s]',
    'CK (median 7d) [U/L]',
    'PTT (max 7d) [s]',
    'Body core temperature (min 4h) [°C]',
    'Leucocytes (trend per day 3d) [thousand/μL]',
    'RHb (median 12h)',
    'Age (static all data) [years] x BE (iqr 3d) [mmol/L]',
    'Hematocrit (max 3d) [%] x Blood volume out (extrapolate 3d) [mL]',
    'Drugs for constipation (unique 1d) x Leucocytes (median 1d) [thousand/μL]',
    'Blood volume out (extrapolate 7d) [mL] x Procalcitonin (max 7d) [ng/mL]',
    'Leucocytes (median 1d) [thousand/μL] x Blood volume out (extrapolate 3d) [mL]'
]


def main():
    parser = argparse.ArgumentParser(description='1. experiment: LR and GAM parameter tuning and rf/feat selection.')
    parser.add_argument('password', type=str, help='Database password.')
    parser.add_argument('item_overview', type=str, help='Description of all PDMS items and generated variables.')
    parser.add_argument('--data', type=str, help='CSV file containing features and targets.')
    parser.add_argument('--figures_output_path', type=str, help='Directory where generated figures will be stored.')
    parser.add_argument('--model', type=str, help='Model used for parameter tuning (lr/ebm).')
    parser.add_argument('--procedure', type=str, help='Parameter tuning or feature/rf selection (pt/sel).')
    parser.add_argument('--reduced_features', action='store_true', help='Use reduced set of features.')
    parser.add_argument('--selected_risk_functions', action='store_true', help='Use selected risk functions for ebm.')
    parser.add_argument('--bin_size', type=str, help='Bin sizes for parameter tuning of ebm.')
    args, _ = parser.parse_known_args()

    # Parameters and data.
    feature_columns, year_column, label_column, data = read_features_years_labels(args.data)
    assert args.model == 'lr' or args.model == 'ebm'
    assert args.procedure in ['1-parameter-tuning',
                              '2-determine-80-most-important-features',
                              '3-risk-function-and-feature-selection',
                              '4-add-2d-risk-functions',
                              '5-plot-relevance-risk-function-and-features']

    # Adjust features when using logistic regression.
    assert len(feature_columns) == 1423
    # Check that there are no duplicated features.
    assert len(feature_columns) == len(set(feature_columns))
    if args.model == 'lr':
        feature_columns, data = add_categorical_dummies_and_nan_indicators(args.item_overview, feature_columns, data)
        assert len(feature_columns) == 1423 + 106 - 39 + 856  # Change 39 feat to dummies, 856 nan indicators
    splits = create_splits(data, feature_columns, year_column, label_column)

    # Consider a reduced set of 130/80 features / risk functions and re-order them to get exact same results as in PT.
    if args.reduced_features:
        if args.model == 'lr':
            removed_feature_columns = [f for f in feature_columns if f not in most_important_130_feat]
            data.drop(removed_feature_columns, axis=1, inplace=True)
            data = data[most_important_130_feat + [year_column] + [label_column]]
            feature_columns = most_important_130_feat
        else:
            removed_feature_columns = [f for f in feature_columns if f not in most_important_80_rf]
            data.drop(removed_feature_columns, axis=1, inplace=True)
            data = data[most_important_80_rf + [year_column] + [label_column]]
            feature_columns = most_important_80_rf
        assert len(feature_columns) == len(data.columns.tolist()) - 2
        assert (args.model == 'lr' and len(feature_columns) == 130) or\
            (args.model == 'ebm' and len(feature_columns) == 80)

    # Consider a selected set of risk functions and re-order them to get exact same results as in PT.
    if args.selected_risk_functions:
        assert args.model == 'ebm'
        removed_feature_columns = [f for f in feature_columns if f not in best_1d_ebm_risk_functions]
        data.drop(removed_feature_columns, axis=1, inplace=True)
        data = data[best_1d_ebm_risk_functions + [year_column] + [label_column]]
        feature_columns = best_1d_ebm_risk_functions
        assert len(feature_columns) == len(data.columns.tolist()) - 2
        assert len(feature_columns) == 80

    if args.procedure == '1-parameter-tuning':
        train_x, train_y, valid_x, valid_y, test_x, test_y = \
            get_split(splits.iloc[-1], data, feature_columns, year_column, label_column)
        # Z-normalize data for LR.
        if args.model == 'lr':
            train_x, valid_x, test_x = z_normalize_based_on_train(train_x, valid_x, test_x)
        print(f"Perform parameter tuning with {len(feature_columns)} features.")

        @output_wrapper
        def model_run(args_model, params):
            start = time.time()
            if args_model == 'lr':
                run_clf = LogisticRegression(verbose=0)
            else:
                run_clf = ExplainableBoostingClassifier()
            run_clf.set_params(**params)
            run_clf.fit(train_x, train_y)

            eval_scores = evaluate_model(run_clf, train_x, train_y, valid_x, valid_y)
            # Return dictionary of all parameters and scores.
            result = {**params, **eval_scores}
            if args.model == 'lr':
                result['n_iter_performed'] = run_clf.n_iter_[0]
            print('\t', datetime.now().strftime("%m/%d/%Y-%H:%M:%S"), f"({(time.time() - start):.2f}s):", result)
            return result

        if args.model == 'lr':
            global lr_params
            print(f"Test {len(ParameterGrid(lr_params))} parameter settings for logistic regression.")
            params_and_scores =\
                Parallel(n_jobs=40)(delayed(model_run)(args.model, params) for params in ParameterGrid(lr_params))
        else:
            # Enable Logging of EBM framework.
            # enable_ebm_logging()
            global ebm_params
            print(f"Test {len(ParameterGrid(ebm_params))} parameter settings for ebm.")
            params_and_scores =\
                Parallel(n_jobs=38)(delayed(model_run)(args.model, params) for params in ParameterGrid(ebm_params))

        out = pd.DataFrame(params_and_scores)
        out.to_csv(f"/home/stefanhgm/MLUKM/Experiments/1_EBM_LR_risk_function_and_feature_selection/"
                   f"{args.model}_parameter-tuning-{str(time.strftime('%Y%m%d-%H%M%S'))}.csv", sep='\t', index=False)

    elif args.procedure == '2-determine-80-most-important-features':
        train_x, train_y, valid_x, valid_y, test_x, test_y = \
            get_split(splits.iloc[-1], data, feature_columns, year_column, label_column)
        # Z-normalize data for LR.
        if args.model == 'lr':
            train_x, valid_x, test_x = z_normalize_based_on_train(train_x, valid_x, test_x)

        if args.model == 'lr':
            model = LogisticRegression(verbose=0)
            model.set_params(**best_lr_params_full_features)
        else:
            model = ExplainableBoostingClassifier()
            model.set_params(**best_ebm_params_full_features)
        model.fit(train_x, train_y)
        scores = evaluate_model(model, train_x, train_y, valid_x, valid_y)
        print(f"Verify performance of parameter tuning - trained full model with best parameters: {scores}")

        # Now select 80 most important risk functions/features but not with temporal split but random split.
        train_valid_x = pd.concat([train_x, valid_x], ignore_index=True)
        train_valid_y = pd.concat([train_y, valid_y], ignore_index=True)
        group_train_x, group_valid_x, group_train_y, group_valid_y =\
            train_test_split(train_valid_x, train_valid_y, test_size=0.15, random_state=seed)
        if args.model == 'lr':
            model = LogisticRegression(verbose=0)
            model.set_params(**best_lr_params_full_features)
        else:
            model = ExplainableBoostingClassifier()
            model.set_params(**best_ebm_params_full_features)
        model.fit(group_train_x, group_train_y)
        scores = evaluate_model(model, group_train_x, group_train_y, group_valid_x, group_valid_y)
        print(f"Performance for grouped train-valid with random split: {scores}")
        # Sub sampled grouped valid splits for importance.
        neg_ss_valid_x, neg_ss_valid_y = create_neg_sub_sampled(group_valid_x, group_valid_y)

        feat_importance = {}
        if args.model == 'lr':
            assert model.coef_.shape[1] == len(feature_columns)
            for idx, feature in enumerate([f for f in feature_columns]):
                coefficient = np.abs(model.coef_[0, idx])
                feat_importance[feature] = np.mean(np.abs(coefficient * neg_ss_valid_x[feature]))
        else:
            importance = manual_feature_importance(model, neg_ss_valid_x, neg_ss_valid_y)
            for feature in [f for f in feature_columns]:
                feat_idx = model.feature_names.index(feature)
                feat_group_idx = model.feature_groups_.index([feat_idx])
                feat_importance[feature] = importance[feat_group_idx]
                assert feature == model.feature_names[feat_group_idx]
        feat_importance = {k: v for k, v in sorted(feat_importance.items(), key=lambda item: item[1], reverse=True)}
        if args.model == 'lr':
            most_80_important = [k for k, v in feat_importance.items()][:130]
        else:
            most_80_important = [k for k, v in feat_importance.items()][:80]
        print(f"Determined 80 most important features/rf on negative sub sampled data.")
        print(most_80_important)

    elif args.procedure == '3-risk-function-and-feature-selection':
        # Stepwise forward selection of features based on maximum improvement of pr_valid on temporal splits.
        if args.model == 'lr':
            parameters = best_lr_params_130_features
            max_features_selected = 160
        else:
            if args.bin_size == '100':
                parameters = best_ebm_params_80_risk_functions_bin_size_100
            elif args.bin_size == '150':
                parameters = best_ebm_params_80_risk_functions_bin_size_150
            elif args.bin_size == '200':
                parameters = best_ebm_params_80_risk_functions_bin_size_200
            elif args.bin_size == '250':
                parameters = best_ebm_params_80_risk_functions_bin_size_250
            elif args.bin_size == '300':
                parameters = best_ebm_params_80_risk_functions_bin_size_300
            elif args.bin_size == '350':
                parameters = best_ebm_params_80_risk_functions_bin_size_350
            elif args.bin_size == '400':
                parameters = best_ebm_params_80_risk_functions_bin_size_400
            else:
                raise ValueError('Invalid bin size.')
            max_features_selected = 100
            print(f"Do EBM risk function selection with parameters: {parameters}")

        @output_wrapper
        def model_run(args_model, feature_name, reduced_train_x, reduced_train_y, reduced_valid_x, reduced_valid_y):
            if args_model == 'lr':
                run_clf = LogisticRegression(verbose=0)
            else:
                run_clf = ExplainableBoostingClassifier()
            run_clf.set_params(**parameters)
            run_clf.fit(reduced_train_x, reduced_train_y)

            eval_scores = evaluate_model(run_clf, reduced_train_x, reduced_train_y, reduced_valid_x, reduced_valid_y)
            # Return dictionary of all parameters and scores.
            # Importance determined on negative sub-sampled sets.
            neg_ss_reduced_valid_x, neg_ss_reduced_valid_y = create_neg_sub_sampled(reduced_valid_x, reduced_valid_y)
            if args.model == 'lr':
                feat_coefficient = np.abs(run_clf.coef_[0, -1])
                ss_importance = (np.mean(np.abs(feat_coefficient * neg_ss_reduced_valid_x.iloc[:, -1:].to_numpy())))
            else:
                ss_importance = manual_feature_importance(run_clf, neg_ss_reduced_valid_x, neg_ss_reduced_valid_y)[-1]
            result = {'feature': feature_name, 'importance': ss_importance, **eval_scores}
            return result

        selected_features = []
        pr_valid_scores = []
        while len(selected_features) < max_features_selected:
            rf1d_improvement = dict()
            rf1d_improvement_list = dict()
            for i in range(0, 5):
                train_x, train_y, valid_x, valid_y, test_x, test_y = \
                    get_split(splits.iloc[i], data, feature_columns, year_column, label_column)
                if args.model == 'lr':
                    train_x, valid_x, test_x = z_normalize_based_on_train(train_x, valid_x, test_x)

                train_valid_x = pd.concat([train_x, valid_x], ignore_index=True)
                train_valid_y = pd.concat([train_y, valid_y], ignore_index=True)
                group_train_x, group_valid_x, group_train_y, group_valid_y =\
                    train_test_split(train_valid_x, train_valid_y, test_size=0.15, random_state=seed)

                # Create all data subsets for all feature selections.
                run_configurations = []
                for feature in [f for f in feature_columns if f not in selected_features]:
                    run_configurations.append((feature, group_train_x[selected_features + [feature]],
                                               group_valid_x[selected_features + [feature]]))
                if args.model == 'lr':
                    feat_and_scores = \
                        Parallel(n_jobs=36)(delayed(model_run)(args.model, feature, group_train_x, group_train_y,
                                                               group_valid_x, group_valid_y)
                                            for (feature, group_train_x, group_valid_x) in run_configurations)
                else:
                    feat_and_scores = \
                        Parallel(n_jobs=22)(delayed(model_run)(args.model, feature, group_train_x, group_train_y,
                                                               group_valid_x, group_valid_y)
                                            for (feature, group_train_x, group_valid_x) in run_configurations)
                feat_and_scores = pd.DataFrame(feat_and_scores)
                for idx, _ in feat_and_scores.iterrows():
                    feat = feat_and_scores.loc[idx, 'feature']
                    rf1d_improvement[feat] =\
                        rf1d_improvement.get(feat, 0) + feat_and_scores.loc[idx, 'importance']
                    rf1d_improvement_list[feat] =\
                        rf1d_improvement_list.get(feat, []) + [feat_and_scores.loc[idx, 'importance']]
                    # Alternative selection based on pr_valid
                    # rf1d_improvement[feat] = rf1d_improvement.get(feat, 0) + feat_and_scores.loc[idx, 'pr_valid']
                    # rf1d_improvement_list[feat] =\
                    #    rf1d_improvement_list.get(feat, []) + [feat_and_scores.loc[idx, 'pr_valid']]
            feat_max_improvement = max(rf1d_improvement, key=rf1d_improvement.get)
            improvement = rf1d_improvement[feat_max_improvement] / 5.
            improvement_sd = np.std(np.array(rf1d_improvement_list[feat_max_improvement]))
            selected_features.append(feat_max_improvement)

            # Evaluate on final validation split.
            train_x, train_y, valid_x, valid_y, test_x, test_y = \
                get_split(splits.iloc[-1], data, feature_columns, year_column, label_column)
            if args.model == 'lr':
                train_x, valid_x, test_x = z_normalize_based_on_train(train_x, valid_x, test_x)
            train_x = train_x[selected_features]
            valid_x = valid_x[selected_features]
            if args.model == 'lr':
                clf = LogisticRegression(verbose=0)
            else:
                clf = ExplainableBoostingClassifier()
            clf.set_params(**parameters)
            clf.fit(train_x, train_y)
            scores = evaluate_model(clf, train_x, train_y, valid_x, valid_y)
            pr_valid_scores.append(scores['pr_valid'])
            print(f"Selected {len(selected_features)}. feature: "
                  f"mean importance on temp splits {improvement:.5f}±{improvement_sd:.4f}, "
                  f"PR (train) {scores['pr_train']:.5f}, "
                  f"PR (valid) {scores['pr_valid']:.5f}; "
                  f"feature: {feat_max_improvement}. "
                  f"({selected_features}, {pr_valid_scores})")

    elif args.procedure == '4-add-2d-risk-functions':
        # Determine mask for included 1D rf.
        mask_1d_rf = [feature_columns.index(rf) for rf in best_1d_ebm_risk_functions]
        max_2d_rf = 20
        # Note that 2D (interaction) bin sizes was set by hand for experiments. Used values 4, 6, 8, 10, 12, 14.
        max_interaction_bins = 4

        # Test all possible 2d functions.
        list_2d_rfs = list(itertools.combinations(range(0, len(best_1d_ebm_risk_functions)), 2))
        assert len(list_2d_rfs) == (len(best_1d_ebm_risk_functions) * (len(best_1d_ebm_risk_functions) - 1)) / 2
        assert len(list_2d_rfs) == 3160
        # Consecutively add 2D risk functions and check the validation performance.
        train_x, train_y, valid_x, valid_y, test_x, test_y = \
            get_split(splits.iloc[-1], data, feature_columns, year_column, label_column)
        clf = ExplainableBoostingClassifier()
        clf.set_params(**{'mains': mask_1d_rf, **best_ebm_params_1d_model})
        clf.fit(train_x, train_y)
        scores = evaluate_model(clf, train_x, train_y, valid_x, valid_y)
        print(f"Before selecting 2D rf: PR (train) {scores['pr_train']:.5f}, PR (valid) {scores['pr_valid']:.5f}"
              f" using {len(best_1d_ebm_risk_functions)} 1D rfs and {max_interaction_bins} max 2D bins.")

        @output_wrapper
        def ebm_2d_run_run(interaction, params, data_train_x, data_train_y, data_valid_x, data_valid_y):
            run_clf = ExplainableBoostingClassifier()
            run_clf.set_params(**params)
            run_clf.fit(data_train_x, data_train_y)

            eval_scores = evaluate_model(run_clf, data_train_x, data_train_y, data_valid_x, data_valid_y)
            # Return dictionary of all parameters and scores.
            # Importance determined on negative sub-sampled sets.
            neg_ss_reduced_valid_x, neg_ss_reduced_valid_y = create_neg_sub_sampled(data_valid_x, data_valid_y)
            ss_importance = manual_feature_importance(run_clf, neg_ss_reduced_valid_x, neg_ss_reduced_valid_y, True)[-1]
            result = {'interaction': interaction, 'importance': ss_importance, **eval_scores}
            return result

        selected_2d_rf = []
        pr_valid_scores = []
        while len(selected_2d_rf) < max_2d_rf:
            rf1d_improvement = dict()
            for i in range(0, 5):
                train_x, train_y, valid_x, valid_y, test_x, test_y = \
                    get_split(splits.iloc[i], data, feature_columns, year_column, label_column)
                train_valid_x = pd.concat([train_x, valid_x], ignore_index=True)
                train_valid_y = pd.concat([train_y, valid_y], ignore_index=True)
                group_train_x, group_valid_x, group_train_y, group_valid_y =\
                    train_test_split(train_valid_x, train_valid_y, test_size=0.15, random_state=seed)

                # Now train models for each 2d rf candidate with already selected 2d rfs and save resp. improvement.
                rf2d_and_scores = \
                    Parallel(n_jobs=8)(delayed(ebm_2d_run_run)
                                        (rf_2d, {**best_ebm_params_1d_model, 'interactions': selected_2d_rf + [rf_2d],
                                                 'mains': mask_1d_rf, 'max_interaction_bins': max_interaction_bins},
                                         group_train_x, group_train_y, group_valid_x, group_valid_y)
                                        for rf_2d in list_2d_rfs if rf_2d not in selected_2d_rf)
                rf2d_and_scores = pd.DataFrame(rf2d_and_scores)
                for idx, _ in rf2d_and_scores.iterrows():
                    rf2d = rf2d_and_scores.loc[idx, 'interaction']
                    rf1d_improvement[rf2d] = rf1d_improvement.get(rf2d, 0) + rf2d_and_scores.loc[idx, 'importance']

            rd2f_max_improvement = max(rf1d_improvement, key=rf1d_improvement.get)
            improvement = rf1d_improvement[rd2f_max_improvement] / 5.
            selected_2d_rf.append(rd2f_max_improvement)

            # Evaluate on final validation split.
            train_x, train_y, valid_x, valid_y, test_x, test_y = \
                get_split(splits.iloc[-1], data, feature_columns, year_column, label_column)
            clf = ExplainableBoostingClassifier()
            clf.set_params(**{**best_ebm_params_1d_model, 'interactions': selected_2d_rf, 'mains': mask_1d_rf,
                              'max_interaction_bins': max_interaction_bins})
            clf.fit(train_x, train_y)
            scores = evaluate_model(clf, train_x, train_y, valid_x, valid_y)
            pr_valid_scores.append(scores['pr_valid'])
            print(rd2f_max_improvement)
            print(f"Selected {len(selected_2d_rf)}. 2d rf: "
                  f"mean importance on temp splits {improvement:.5f}, "
                  f"PR (train) {scores['pr_train']:.5f}, "
                  f"PR (valid) {scores['pr_valid']:.5f}; "
                  f"interaction: {rd2f_max_improvement} "
                  f"({feature_columns[rd2f_max_improvement[0]]} x {feature_columns[rd2f_max_improvement[1]]}) "
                  f"({selected_2d_rf}, {pr_valid_scores})")

    elif args.procedure == '5-plot-relevance-risk-function-and-features':
        train_x, train_y, valid_x, valid_y, test_x, test_y = \
                get_split(splits.iloc[-1], data, feature_columns, year_column, label_column)
        pr_valid_baseline = np.sum(valid_y) / valid_y.shape[0]
        ebm_pr_valid_rf_selection_per_bin_size = \
            [[0.042198496118159956, 0.05536286788102365, 0.058095279291340576, 0.059114182698213405, 0.05814923545264762,
             0.05991932507100133, 0.07360460682455833, 0.07178793511719675, 0.07373662267651879, 0.07333822099842591,
             0.07496065856373775, 0.0727609695679709, 0.07527540365327297, 0.07714957621370262, 0.07779373345194877,
             0.08042773684623512, 0.08254014210078603, 0.08300371321494353, 0.08125894388415822, 0.08190525126193472,
             0.07163175529613644, 0.07892288785717866, 0.07402802473254189, 0.09132388683415583, 0.08663937582842081,
             0.08832070734733405, 0.07517063706547251, 0.07613784888593322, 0.0751708889854586, 0.07631630580204973,
             0.07326722451469264, 0.0736586999273971, 0.07379092889156366, 0.0758061432633375, 0.09176717336652121,
             0.08765193402077266, 0.08956983748316473, 0.08621152543737307, 0.07577855311668659, 0.07294627888055315,
             0.07181700807823171, 0.07097946149422883, 0.0734376571004455, 0.08441589030975707, 0.06748822620521006,
             0.06745960457785904, 0.06790693523910864, 0.06631999083058567, 0.06830226765846094, 0.08219935947722708,
             0.06982249207674025, 0.06676454184696935, 0.06816992829012916, 0.06805271352962239, 0.07190008965308617,
             0.06813597009051717, 0.06919226589642631, 0.0704046341387884, 0.0714382863391094, 0.07301762023076436,
             0.07597529719833224, 0.07516773915673801, 0.08546647924431224, 0.07531283683926396, 0.07285952513488779,
             0.0750080114214747, 0.07657220654521975, 0.0735159649854012, 0.07393603541164398, 0.07270077206716724,
             0.07316608909256087, 0.07445363113253212, 0.07562964508982477, 0.0771553053863077, 0.0800076900053727,
             0.07741337740802008, 0.07637949538850627, 0.07591710313525414, 0.08144442272321303, 0.08429005463677389],
            [0.05272339326309752, 0.06931101884808627, 0.059260986745748014, 0.059347831302401365, 0.06151793177742664,
             0.06355944207412811, 0.08204491581702536, 0.07195537591480299, 0.06564113897425664, 0.067610567231424,
             0.06617037352100133, 0.06828344558343319, 0.06700596877752447, 0.06708707026968985, 0.06532371540291088,
             0.06584350376570001, 0.06831315696462681, 0.06507258277122066, 0.06164778548537568, 0.060266472266317095,
             0.05938004214919484, 0.06092435675430781, 0.06267148433126482, 0.06370562411377403, 0.06220995653080162,
             0.06488704646465561, 0.06407771497978779, 0.0650399629825715, 0.06452093000639246, 0.06454091373616329,
             0.06759288328828697, 0.0670700153608984, 0.06789358552796548, 0.07045575247404125, 0.07327950843906722,
             0.07127854657659369, 0.07240326175273792, 0.07228909897477645, 0.07402890637923065, 0.07480164499486217,
             0.07257801973605246, 0.07360455811835792, 0.07507010477809983, 0.07534194689809934, 0.07679476050124813,
             0.0750353601017682, 0.07503329760517399, 0.07623832785430934, 0.07577830851543302, 0.07575299602139102,
             0.0772909790226915, 0.0767073885644239, 0.07675772071046036, 0.07868658113461689, 0.08040886253594501,
             0.07872303266639286, 0.07800028342997699, 0.07970302697056489, 0.0796580828338905, 0.07766769655156205,
             0.07990904401314014, 0.08022048330459165, 0.07870052106401824, 0.08060422834586792, 0.0832267456192042,
             0.08465572420254744, 0.08542050429132951, 0.08512409094667293, 0.08493789113333584, 0.08749636735833856,
             0.08579441677594367, 0.0860591513923464, 0.08568544833681685, 0.08564331027419025, 0.08733862937617869,
             0.08584894534186605, 0.0885470924995323, 0.08919763401696706, 0.08571421614693267, 0.08495965147522377],
            [0.054333702452143785, 0.055107356957992154, 0.06101968493910807, 0.06040077543907993, 0.06074131036911371,
             0.06274424923456405, 0.06340056704314269, 0.06515669959025468, 0.07919681669648607, 0.07940087390410962,
             0.07808296175693924, 0.07936342560237346, 0.07885800130628198, 0.08021788983338755, 0.07885250535286395,
             0.06678588820764123, 0.0685401688151052, 0.06788416902890965, 0.06957491041170212, 0.07641604542336954,
             0.07345491218290331, 0.07744291342664458, 0.07980272750350292, 0.07792596347485277, 0.0777871693515854,
             0.07913913095767268, 0.08016725528435288, 0.08310798141960277, 0.08689619350104039, 0.0845867144811852,
             0.086455408947141, 0.08703773414386373, 0.08248566141712914, 0.08332687066775425, 0.08231014073466694,
             0.08103728993388658, 0.08075679901909094, 0.08256043791420567, 0.0832469164286384, 0.08326832203562891,
             0.08553894317779534, 0.0813466052631568, 0.08168123862462476, 0.08012215026955201, 0.08014406333670285,
             0.0795765078372221, 0.07685816078275921, 0.08062291572639473, 0.0793968071348281, 0.08428893239169791,
             0.08316956957731511, 0.0797054170631066, 0.08338928606128758, 0.07759548427186817, 0.07774307718149207,
             0.0838635618726143, 0.08006193991264297, 0.07474961068063489, 0.07552203844258615, 0.076779490837221,
             0.074716579783644, 0.07722952661166801, 0.07834500824567991, 0.07897539354969371, 0.0798272324247284,
             0.08100006957691341, 0.08098797874686328, 0.0824817645292564, 0.0807204584132569, 0.0811123277039604,
             0.08120424300293609, 0.08360286566385389, 0.08109900978050766, 0.07830264846931298, 0.0801766600842751,
             0.08141125053287651, 0.08073965470122366, 0.0815884071420492, 0.09546354406317537, 0.09641185980558956],
            [0.25988966432584665, 0.06337687499861058, 0.060905707449564084, 0.056998563000757785, 0.055915004586253855,
             0.05659558058931092, 0.056609343965165045, 0.06260864778980751, 0.06450426526509014, 0.0645002486914387,
             0.0683388453669456, 0.06812955213854754, 0.07133510477630842, 0.06943482436622372, 0.06589857441610475,
             0.06276873602680415, 0.06861656559639794, 0.08763143692461683, 0.08747024650157192, 0.0864690669926637,
             0.07272044970867596, 0.07521729181633972, 0.0735357841008027, 0.07405881007565618, 0.07255864243270715,
             0.07387905841372144, 0.07176063346266287, 0.07206345877010445, 0.07133506917555035, 0.0746077851549895,
             0.07267915079952822, 0.07394994295797364, 0.07543595247594082, 0.07273000918080078, 0.07336525685245103,
             0.07296540124078114, 0.06693738568233917, 0.06308963248723114, 0.061796028276405214, 0.0633565009743836,
             0.06429131569639322, 0.06222711747204811, 0.06225981774511344, 0.07954549076393494, 0.06825424207783827,
             0.07927031349399113, 0.06460793563534092, 0.06124009478478329, 0.06502676474316815, 0.0650172956569656,
             0.0675472900304292, 0.06489465736510801, 0.0625218271461081, 0.06278644093381044, 0.06196106792105191,
             0.06187089349791966, 0.06302820484389766, 0.06100918883951336, 0.06063811500323489, 0.06016770952381267,
             0.05932288000316319, 0.05980983441467023, 0.06040097786258539, 0.061958545938658904, 0.06153179524216407,
             0.062276607147811774, 0.06241711848168636, 0.06261413930093113, 0.06174556509963905, 0.06064109280235018,
             0.06258508775430656, 0.062094380034933304, 0.06167421521482304, 0.06162161550095143, 0.06150887710711933,
             0.062007622717592026, 0.0619575245827033, 0.06218991226576018, 0.06401081523965958, 0.06339739930753358],
            [0.04109555216931251, 0.05455669566164481, 0.057645967043585, 0.059570989389017344, 0.05624131308045872,
             0.055867615648744276, 0.05835285041037308, 0.0698270784427588, 0.0689638555467122, 0.06530489886830397,
             0.0648223725739557, 0.06347587144056019, 0.06311017467056888, 0.0625976605046409, 0.06135491139630692,
             0.059456723079476345, 0.05930876798405085, 0.05900063267734868, 0.06120755033948207, 0.06047689474013304,
             0.059126593503551136, 0.06147474940717202, 0.06385657717144036, 0.06123331530859986, 0.06508673699394596,
             0.06411582851012891, 0.060785489887329555, 0.060711785948429235, 0.06619084291687301, 0.0688150731421637,
             0.06858520512847692, 0.06843048113874918, 0.06835026374108238, 0.07004174407511864, 0.07232301005221568,
             0.0736558053599966, 0.07260631707887982, 0.07123591990898252, 0.07164823446811742, 0.072689515696889,
             0.07077816179247493, 0.07128522564003183, 0.07135717165358274, 0.07108202197247315, 0.07131000850452096,
             0.07323570041628254, 0.07192338659009948, 0.07223604014954257, 0.07340624180093605, 0.07166593036431367,
             0.07081909143600745, 0.07138939815467917, 0.07232607754107202, 0.07096127256238104, 0.07243935446675154,
             0.07137200433819592, 0.07077753672944104, 0.07046962532033216, 0.07073442217741928, 0.07154564431432153,
             0.07135623358230289, 0.07233628432372807, 0.0719493442780886, 0.07166085822791385, 0.07118015493964946,
             0.0710094864122904, 0.07341468372449271, 0.07010325052478103, 0.06969179430622018, 0.07118896003740158,
             0.06971560204135495, 0.07186605964686905, 0.07152889184994238, 0.07412598784582661, 0.07378286770346604,
             0.07400564516500652, 0.07376419385578392, 0.07392728334468882, 0.07494514778824685, 0.07158235382577285],
            [0.07724860416927135, 0.059178940903767414, 0.05718478823517788, 0.05985141012406467, 0.0575908466289051,
             0.05723951787044956, 0.05939684978138113, 0.06161078815147423, 0.06173751089235777, 0.06052745875737744,
             0.058330221812022895, 0.059451123840502496, 0.06038484755201702, 0.0619780569328375, 0.062447617244336576,
             0.06046264479265201, 0.06101462310681543, 0.060941530337339, 0.06251147044100575, 0.060930699885071195,
             0.061205239812123835, 0.06073220109953825, 0.06184488949298855, 0.0611448138981507, 0.063212342200633,
             0.0614009356290753, 0.06329956472980959, 0.06291573026830258, 0.06579876180215474, 0.06474506666572824,
             0.06689416258753599, 0.06892334442358068, 0.07063104073832205, 0.07045000459357245, 0.06992113020176005,
             0.0723452065080911, 0.07176941124049407, 0.0728165285017184, 0.07159188695649915, 0.07200188933568805,
             0.07345300687310702, 0.07178819578300344, 0.06976438054123246, 0.07247418523673244, 0.071406848918048,
             0.07064670576180725, 0.0704719321877853, 0.07101435959186585, 0.07011938428255865, 0.0682306639688654,
             0.06937672861143998, 0.06921290251045992, 0.07129905740362523, 0.07248530512173199, 0.07269697935211687,
             0.07347294084828451, 0.07347777464762048, 0.07176529819575811, 0.07421107523527076, 0.07409702878654198,
             0.07697729315909892, 0.0773508439095327, 0.07525741795610892, 0.07577861147487487, 0.07509249964540057,
             0.07366750119582144, 0.0747723159335845, 0.07383062087125086, 0.07599305568153894, 0.07401588115602178,
             0.07242706685231529, 0.0728174437762083, 0.07319418119301613, 0.07424531399989542, 0.076055320825311,
             0.0754028465668504, 0.0731738798110719, 0.07276626513912332, 0.07252382517029056, 0.07362112482641231],
            [0.15912149905645984, 0.047062391824491734, 0.04519292654187736, 0.05565192191663408, 0.05466199996011541,
             0.05771765772099103, 0.056812583556072674, 0.0576258735598297, 0.05704785310499521, 0.05614249003766368,
             0.056491224512021015, 0.056855739612284996, 0.05696086925706197, 0.05707154651675071, 0.057338383988397815,
             0.05853057376089496, 0.058243085067892934, 0.05809827414461788, 0.06061987248804619, 0.06060599264533395,
             0.06317715676138225, 0.06157621028140429, 0.06411870844331607, 0.06060490053755953, 0.06198490259052534,
             0.06000407296167783, 0.06154835320089999, 0.06248946139592366, 0.062011643789169235, 0.05983018520540562,
             0.060972949766364845, 0.060437724409604196, 0.058255422549876174, 0.05656148622660629, 0.054050567132205046,
             0.05452064469880825, 0.05493379245962453, 0.055842865905449166, 0.055050794797604295, 0.0542141157503659,
             0.052391530546530576, 0.05307785765170574, 0.05319652897097512, 0.05472564694090647, 0.053622692341139366,
             0.053570219950297544, 0.055744548012480424, 0.05493895329223635, 0.05537397201904236, 0.05396122987437361,
             0.05456243534206377, 0.054989642769227215, 0.05606725532243169, 0.056342740294467133, 0.05459027216579002,
             0.05516502109319357, 0.05353852506623018, 0.056264407222662564, 0.056528919694286923, 0.05684848753819531,
             0.05768646800053767, 0.058944680971144144, 0.057806860703130784, 0.055583671310886, 0.05641829494913427,
             0.056435147872769154, 0.05587106179595014, 0.05553385788155587, 0.05491271849715221, 0.053295903574656564,
             0.05405604542040444, 0.05577315563564762, 0.05785525279367229, 0.05503170117473373, 0.056935213686233416,
             0.05306865111473208, 0.05212291662121304, 0.05384465888442255, 0.053522709136330116, 0.05451716269056216]]
        ebm_pr_valid_rf_2d_selection_per_bin_size = [
            [0.09975994603483887, 0.09404915154936049, 0.09686209727759845, 0.0969343748719374, 0.10103192417471156,
             0.09358117073898117, 0.09449550974369271, 0.09483975128910745, 0.09543452782020748, 0.09376025213529407,
             0.09051418811526182, 0.09136106227377867, 0.09071743058590026, 0.09282150454332874, 0.09243259742662831,
             0.09175942362652201, 0.09386718321074045, 0.09233597173122485, 0.09253055101986149, 0.09162958934680417],
            [0.0992150591973939, 0.09779714748325385, 0.0981801303835301, 0.09669772861563977, 0.0952021946670634,
             0.09591407213169012, 0.09914127680758943, 0.09759173808694899, 0.09515293953279454, 0.09367987978775977,
             0.09404824779817003, 0.09425528332339721, 0.0934926550973118, 0.07769112498665504, 0.07766140614338565,
             0.07883667598363824, 0.07937633756791762, 0.08173950484631269, 0.07876101990130147, 0.07859748625866818],
            [0.0971138785277042, 0.09747799632259752, 0.09497595837219489, 0.1002260405195202, 0.09661308412913458,
             0.09763335829628075, 0.09485291723452931, 0.09324349858486769, 0.09411337515312987, 0.09374808887315747,
             0.09088371950104683, 0.0912982312499071, 0.08924241943075059, 0.0886116685450674, 0.08845864055321605,
             0.0902023986171875, 0.09076977071428327, 0.09241857979132408, 0.07919887182187352, 0.09245284183892032],
            [0.09792670350917798, 0.0976451298954889, 0.09687412610763327, 0.09529320566463066, 0.094889766393468,
             0.09540492305657243, 0.09488393162607084, 0.09851245591935964, 0.10067068078421246, 0.10022358090402174,
             0.0967922208349742, 0.09589227335135603, 0.09113180030870249, 0.09011595331581333, 0.08928037770311285,
             0.09048127850867073, 0.08770942531581327, 0.08763142607652481, 0.08607001270445197, 0.08669206357088245],
            [0.08628434063589309, 0.08675945087867559, 0.08995023922356263, 0.09092638124219207, 0.08868696315588195,
             0.08887935313876771, 0.09060166753541322, 0.09615377717654823, 0.09373622676738977, 0.09382720616654099,
             0.09142749445376666, 0.09326645668033084, 0.09198860562740875, 0.0916548385506282, 0.09200645627496712,
             0.09200665902023478, 0.09038011055930424, 0.08567719399934193, 0.08620556197684072, 0.08598156831991077],
            [0.09643197939685141, 0.09691588791039668, 0.09528770184871131, 0.09787404680974814, 0.09582021762735285,
             0.09704630029107622, 0.09835088789180892, 0.0890557488618423, 0.09234147741858087, 0.09248217770273871,
             0.09167693255549106, 0.09338744255677953, 0.09400313478475326, 0.0934650609470144, 0.09246917494008576,
             0.09174601445946659, 0.09094099630484206, 0.09034434226820562, 0.09307932907345956, 0.09150142590804364]]
        lr_pr_valid_features_selection = \
            [0.1128552397432312, 0.05152915506193531, 0.0547463132952842, 0.06912814998964065, 0.06135493200784571,
             0.05835972813809083, 0.057224639843036586, 0.07102072394438276, 0.07227432099109588, 0.07228025056817927,
             0.05668560146537848, 0.056461296139496256, 0.06924538658373544, 0.06876208835452882, 0.06881804030765286,
             0.06890002654025507, 0.06940413997667987, 0.0715231406163614, 0.05698047131726858, 0.05698440628575893,
             0.05698552092258139, 0.05704356650563148, 0.05592242555722704, 0.05515786241674402, 0.05496001763552625,
             0.05528565284654407, 0.05522909938816346, 0.05466091627412441, 0.05127588022432691, 0.05133078888087697,
             0.05216762213658723, 0.05248776935002834, 0.05094866245245784, 0.05094407912582584, 0.05525753359984274,
             0.05623173449616722, 0.05621107445344463, 0.05627970324150887, 0.057022293119485, 0.05542490684905764,
             0.05622600784746831, 0.05586392822502874, 0.05549610695729755, 0.05745355557734105, 0.0573110263928717,
             0.05751606147674636, 0.07213795447042648, 0.07211230581965955, 0.072293985906634, 0.059866838124796165,
             0.058776485811350886, 0.058625565862495725, 0.059453274801066075, 0.06031248007580513, 0.06029170944639106,
             0.0604260511533649, 0.06031926325712367, 0.06052023735097643, 0.060369212174194914, 0.060323320259962905,
             0.06038791262539339, 0.060462454494637094, 0.06092412392909112, 0.06188482402424367, 0.06169586710713057,
             0.06171757614723196, 0.06127783278988079, 0.060887966834501185, 0.06204369087512216, 0.06094784262792961,
             0.061102868126562876, 0.061365089191061935, 0.060961561030315484, 0.06205027846366624, 0.06274771291732717,
             0.06358926520392313, 0.06354598184683252, 0.06350429150498607, 0.0635454104783437, 0.06350336252269306,
             0.06359288012681436, 0.06290182767193372, 0.06291970165014181, 0.06185973211563489, 0.06221716902902316,
             0.062150337887193666, 0.06222861475035857, 0.06204297749812333, 0.062027359061631156, 0.062395346270888996,
             0.06236242145230056, 0.062388424359805475, 0.06255909885184172, 0.0623461347967611, 0.0623455387400047,
             0.062191499482372, 0.06263629081935793, 0.06274087678012535, 0.06290861324373864, 0.06325869737039783,
             0.06175889697289691, 0.06174632587645048, 0.060933674252519235, 0.061632598906161684, 0.06130422690828359,
             0.061530094422592614, 0.06143288216431323, 0.062235620784363045, 0.06159368097122889, 0.061430352664427784,
             0.06176630303392018, 0.061508492500341065, 0.061196453964916776, 0.061383870236635586, 0.06159027936243692,
             0.05987568815752575, 0.061391103232057875, 0.06135773412426903, 0.062492221108911825, 0.06280873205233986,
             0.0671965961771882, 0.06716395442759887, 0.06597923483466603, 0.06621450153355918, 0.0673069685129882,
             0.0671052434072812, 0.0654222124979022, 0.06583332875463577, 0.06524726803885542, 0.0658340942406487]
        for x in ebm_pr_valid_rf_selection_per_bin_size:
            assert len(x) == 80
        assert len(lr_pr_valid_features_selection) == 130
        ebm_2d_start = 80
        ebm_2d_start_score = 0.09641185980558956
        ebm_2d_best_score = 0.10103192417471156

        matplotlib.rcParams.update({'font.size': 12})
        fig, ax = plt.subplots(figsize=(10, 7))
        x_size = 130
        # Plot baseline and pr-valid curves.
        ax.plot([0, x_size], [pr_valid_baseline, pr_valid_baseline], color='black', linestyle='--')
        smoothed_lr_pr_valid_features_selection = pd.Series([pr_valid_baseline] + lr_pr_valid_features_selection)\
            .rolling(window=1, min_periods=1, center=True).mean().values
        ax.plot(range(0, x_size + 1), smoothed_lr_pr_valid_features_selection[0:x_size+1], color='black', label=f"LR")
        cm = plt.cm.get_cmap('tab20').colors
        for i, x in enumerate(ebm_pr_valid_rf_selection_per_bin_size):
            bin_size = ebm_params['min_samples_bin'][i]
            smoothed = (pd.Series([pr_valid_baseline] + x)).rolling(window=1, min_periods=1, center=True).mean().values
            ax.plot(range(0, 81), smoothed, color=cm[i*2], label=f"EBM (1D bin size {bin_size})")
        for i, x in enumerate(ebm_pr_valid_rf_2d_selection_per_bin_size):
            bin_size = 4 + 2*i
            smoothed = (pd.Series([ebm_2d_start_score] + x)).rolling(window=1, min_periods=1, center=True).mean().values
            ax.plot(range(ebm_2d_start, 101), smoothed, color=cm[i*2 + 1],  label=f"EBM (2D bin size {bin_size})")

        # Mark chose model.
        plt.plot(80, ebm_2d_start_score, marker='x', color='black')
        ax.text(79, ebm_2d_start_score, 'EBM chosen from 1D risk function selection', ha='right')
        plt.plot(85, ebm_2d_best_score, marker='x', color='black')
        ax.text(84, ebm_2d_best_score, 'EBM chosen from 1D and 2D risk function selection', ha='right')

        ax.grid()
        ax.set_xlim([0.0, x_size])
        ax.set_ylim([0.00, 0.12])
        ax.set_xlabel('Number of included risk functions (EBM) / features (LR)')
        ax.set_ylabel(f"Precision-recall on full validation set")
        # plot.title('Performance depending on number of risk functions / features')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(os.path.join(args.figures_output_path, 'exp1_relevance_number_rf_pr_valid.png'),
                    dpi=600, bbox_inches='tight')

    return


def create_neg_sub_sampled(x, y):
    sampled_neg_y = y[y == 0].sample(n=y[y == 1].shape[0], replace=False, random_state=seed)
    sub_sampled_neg_y = y[y == 1].append(sampled_neg_y)
    sub_sampled_neg_x = x.loc[sub_sampled_neg_y.index]
    assert sub_sampled_neg_y.shape[0] == 2 * y[y == 1].shape[0] and sub_sampled_neg_x.shape[0] == 2 * y[y == 1].shape[0]
    return sub_sampled_neg_x, sub_sampled_neg_y


def manual_feature_importance(model, x_in, y_in, pairs=None):
    """
    Taken from original EBM code to generate importance manually and for arbitrary inputs.
    """
    x_orig, y_orig, _, _ = unify_data(x_in, y_in, model.feature_names, model.feature_types, missing_data_allowed=True)
    x = model.preprocessor_.transform(x_orig)
    x = np.ascontiguousarray(x.T)
    x_pair = None
    if pairs:
        x_pair = model.pair_preprocessor_.transform(x_orig)
        x_pair = np.ascontiguousarray(x_pair.T)
    # Taken from EBM code.
    scores_gen = EBMUtils.scores_by_feature_group(x, x_pair, model.feature_groups_, model.additive_terms_)
    feature_importance = []
    for set_idx, _, scores in scores_gen:
        mean_abs_score = np.mean(np.abs(scores))
        feature_importance.append(mean_abs_score)
    return feature_importance


def plot_relevance_number_rfs(performance_evaluations, measure, output_path):

    fig, ax = plt.subplots(figsize=(4, 4))

    x_size = len(performance_evaluations[1]) + 1
    ax.plot([0, x_size - 1], [0, 1], color='black', linestyle='--')

    for start, performances, performances_sd, name, colour in performance_evaluations:
        # First calculate AUROC score and SD.
        ax.plot(range(0, x_size), start + performances, color=colour, label=f"{name}")
        ax.fill_between(range(0, x_size),
                        [0] + [a_i - b_i for a_i, b_i in zip(performances, performances_sd)],
                        [0] + [a_i + b_i for a_i, b_i in zip(performances, performances_sd)],
                        color=colour, alpha=0.25)

    ax.grid()
    ax.set_xlim([0.0, x_size - 1])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Number of included risk functions')
    ax.set_ylabel(f"Performance ({measure})")
    # plot.title('Receiver operating characteristic')
    ax.legend(loc="upper left")
    plt.savefig(os.path.join(output_path, 'exp1_relevance_number_rf_' + str(measure).replace(' ', '') + '.png'),
                dpi=600, bbox_inches='tight')


def roc_auc_with_sd(held_out_prob_scores_test, development_prob_scores_test):
    held_out_roc_auc = roc_auc_score(held_out_prob_scores_test[0], held_out_prob_scores_test[1])
    held_out_fpr, held_out_tpr, _ = roc_curve(held_out_prob_scores_test[0], held_out_prob_scores_test[1])
    development_roc_auc = [roc_auc_score(test, prob_scores) for test, prob_scores in development_prob_scores_test]
    sd_roc_auc = np.std(np.array(development_roc_auc), axis=0)
    return held_out_roc_auc, sd_roc_auc, held_out_fpr, held_out_tpr


def plot_roc_auc_with_sd(ax, performance_evaluations):
    # Format of performance_evaluations:
    # [((held_out_labels, held_out_prob), [(split_1_labels, split_1_prob), ... (split_5 ...)], name, colour), ...]

    for held_out_prob_scores_test, development_prob_scores_test, name, colour in performance_evaluations:
        # First calculate AUROC score and SD.
        held_out_roc_auc, sd_roc_auc, held_out_fpr, held_out_tpr =\
            roc_auc_with_sd(held_out_prob_scores_test, development_prob_scores_test)

        # Determine standard deviation over held out splits as measure for the variance.
        development_fpr_tpr = []
        for y_test, prob_scores_test in development_prob_scores_test:
            fpr, tpr, _ = roc_curve(y_test, prob_scores_test)
            development_fpr_tpr.append(pd.DataFrame(np.array([fpr, tpr]).transpose(), columns=['fpr', 'tpr']))
        # Determine sd of tpr over development splits for all fpr values of held out split.
        development_sd_tpr = pd.DataFrame([], columns=['fpr', 'sd_tpr'])
        for fpr in held_out_fpr:
            tpr_values = [((fpr_tpr_split.loc[fpr_tpr_split['fpr'] <= fpr]).iloc[[-1]])['tpr'].iloc[0]
                          for fpr_tpr_split in development_fpr_tpr]
            development_sd_tpr = development_sd_tpr.append({'fpr': fpr, 'sd_tpr': np.std(tpr_values)}, ignore_index=True)

        ax.plot(held_out_fpr, held_out_tpr, color=colour, label=f"{name} ({held_out_roc_auc:.3f}±{sd_roc_auc:.3f})")
        sd_tpr = development_sd_tpr['sd_tpr'].to_numpy()
        ax.fill_between(held_out_fpr, held_out_tpr - sd_tpr, held_out_tpr + sd_tpr, color=colour, alpha=0.25)

    ax.plot([0, 1], [0, 1], color='black', linestyle='--')
    ax.grid()
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('1 - specificity (false positive rate)')
    ax.set_ylabel('sensitivity (true positive rate)')
    ax.legend(loc="lower right")


def pr_auc_score(labels, probabilities):
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    return auc(recall, precision)


def pr_auc_with_sd(held_out_prob_scores_test, development_prob_scores_test):
    held_out_precision, held_out_recall, held_out_thresholds = \
        precision_recall_curve(held_out_prob_scores_test[0], held_out_prob_scores_test[1])
    held_out_pr_auc = auc(held_out_recall, held_out_precision)
    development_precision_recall_thresholds = \
        [precision_recall_curve(test, prob_scores) for test, prob_scores in development_prob_scores_test]
    development_pr_auc = [auc(p_r_t[1], p_r_t[0]) for p_r_t in development_precision_recall_thresholds]
    sd_pr_auc = np.std(np.array(development_pr_auc), axis=0)
    return held_out_pr_auc, sd_pr_auc, held_out_precision, held_out_recall


def plot_pr_auc_with_sd(ax, performance_evaluations):
    # Format of performance_evaluations:
    # [((held_out_labels, held_out_prob), [(split_1_labels, split_1_prob), ... (split_5 ...)], name, colour), ...]

    for held_out_prob_scores_test, development_prob_scores_test, name, colour in performance_evaluations:
        # First calculate PR score and SD.
        held_out_pr_auc, sd_pr_auc, held_out_precision, held_out_recall = \
            pr_auc_with_sd(held_out_prob_scores_test, development_prob_scores_test)

        # Determine standard deviation over held out splits as measure for the variance.
        development_precision_recall = []
        for y_test, prob_scores_test in development_prob_scores_test:
            precision, recall, thresholds = precision_recall_curve(y_test, prob_scores_test)
            development_precision_recall.append(pd.DataFrame(np.array([precision, recall]).transpose(),
                                                             columns=['precision', 'recall']))
        # Determine sd of precision over development splits for all recall values of held out split.
        development_sd_precision = pd.DataFrame([], columns=['recall', 'sd_precision'])
        for recall in held_out_recall:
            # Use index 0 here instead of -1 for ROC-AUC because ordering reversed.
            precision_values = [((pre_rec_split.loc[pre_rec_split['recall'] <= recall]).iloc[[0]])['precision'].iloc[0]
                                for pre_rec_split in development_precision_recall]
            development_sd_precision = development_sd_precision.append(
                {'recall': recall, 'sd_precision': np.std(precision_values)}, ignore_index=True)

        ax.plot(held_out_recall, held_out_precision, color=colour,
                label=f"{name} ({held_out_pr_auc:.3f}±{sd_pr_auc:.3f})")
        sd_precision = development_sd_precision['sd_precision'].to_numpy()
        ax.fill_between(held_out_recall, held_out_precision - sd_precision, held_out_precision + sd_precision,
                        color=colour, alpha=0.25)
        # Output recall, precision pairs with SD.
        # pre_rec_pairs = list(zip(held_out_recall[::-1], held_out_precision[::-1], sd_precision[::-1]))
        # relevant_recall_values = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8]
        # for value in relevant_recall_values:
        #     for rec, pre, sd in pre_rec_pairs:
        #         if rec > value:
        #             print(f"At a recall of {value} a precision of {pre:.3f}±{sd:.3f}.")
        #             break

    # ax.plot([0, 1], [0, 1], color='black', linestyle='--')
    ax.grid()
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('recall')
    ax.set_ylabel('precision (positive predictive value)')
    ax.legend(loc="upper right")


def plot_performance(performance_evaluations, output_path):
    matplotlib.rcParams.update({'font.size': 13})
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(13, 6))
    plot_pr_auc_with_sd(ax_pr, performance_evaluations)
    plot_roc_auc_with_sd(ax_roc, performance_evaluations)
    ax_pr.set_title('(a) Precision-recall curve (PR curve)')
    ax_roc.set_title('(b) Receiver operating characteristic (ROC)')
    plt.savefig(os.path.join(output_path, 'performance_pr_roc.png'), dpi=600, bbox_inches='tight')


def read_data(train_data, valid_data, test_data):
    df_train = pd.read_csv(train_data)
    df_valid = pd.read_csv(valid_data)
    df_test = pd.read_csv(test_data)

    return df_train, df_valid, df_test


def plot_scatter(labels, predictions, plot):
    """ Method to plot a scatter plot of predictions vs labels """
    plot.scatter(labels, predictions)
    plot.xlabel('Labels')
    plot.ylabel('Predictions')
    plot.title('Labels vs predictions')


def add_categorical_dummies_and_nan_indicators(item_overview, feature_columns, data,
                                               add_nan_indicators=True, only_variables=False):
    # Create k-1 dummy variables for k categories of categorical features and
    # replace nan of continuous variables with min-1 and add indicator variable.
    item_processing = read_item_processing_descriptions_from_excel(item_overview)
    categorical_variables = item_processing.loc[item_processing['type'] == 'categorical', 'variable_name'].tolist()
    additional_categorical_variables = ['Is 3-day ICU readmission', 'ICU station']
    categorical_features = ['static', 'last', 'has-true', 'has true', 'has-entry', 'has entry']
    new_feature_columns = []
    num_features_with_dummy = 0
    num_dummy_features = 0
    num_nan_indicators = 0
    for feature_column in feature_columns:
        # Add dummies.
        if (any([(feature_column.startswith(var)) for var in categorical_variables]) and
            (any([('(' + feat in feature_column) for feat in categorical_features]) or only_variables)) or \
                feature_column in additional_categorical_variables:
            one_hot = pd.get_dummies(data[feature_column], prefix=feature_column, dummy_na=True, drop_first=False)
            # Check if nan column (last one) has no values. If so, remove it.
            if str(one_hot.columns[-1]) == feature_column + '_nan' and one_hot.iloc[:, -1].nunique() == 1:
                one_hot = one_hot.iloc[:, :-1]
            # Verify that no duplicates created.
            if len(one_hot.columns.tolist()) != len(set(one_hot.columns.tolist())):
                duplicates = [x for x, count in collections.Counter(one_hot.columns.tolist()).items() if count > 1]
                raise Exception("Create duplicated dummy columns", duplicates)
            data.drop(feature_column, axis=1, inplace=True)
            data = data.join(one_hot)
            new_feature_columns += one_hot.columns.tolist()
            num_features_with_dummy += 1
            num_dummy_features += len(one_hot.columns.tolist())
        else:
            new_feature_columns += [feature_column]
            # Add nan indicators.
            if data[feature_column].isnull().values.any() & add_nan_indicators:
                nan_feature_column = feature_column + '_nan'
                data[nan_feature_column] = 0
                data.loc[data[feature_column].isna(), nan_feature_column] = 1
                min_value = data[feature_column].min()
                data[feature_column] = data[feature_column].fillna(min_value - 1)
                new_feature_columns += [nan_feature_column]
                num_nan_indicators += 1
    assert len(new_feature_columns) == len(feature_columns) + num_dummy_features - num_features_with_dummy + \
           (num_nan_indicators if add_nan_indicators else 0)
    print(f"\tAdded {num_dummy_features} dummies for {num_features_with_dummy} features and "
          f"{num_nan_indicators} nan indicators and replaced the respective nans with min-1.")
    feature_columns = new_feature_columns
    if add_nan_indicators:
        assert not data.isnull().values.any()
    # Check that there are no duplicated features.
    assert len(feature_columns) == len(set(feature_columns))
    return feature_columns, data


def evaluate_model(model, train_x=None, train_y=None, valid_x=None, valid_y=None, test_x=None, test_y=None):
    # Determine all relevant scores for evaluation.
    if model is None:
        prob_train_x = train_x if train_x is not None else None
        prob_valid_x = valid_x if valid_y is not None else None
        prob_test_x = test_x if test_x is not None else None
    elif isinstance(model, keras.Model):
        prob_train_x = model.predict(train_x) if train_x is not None else None
        prob_valid_x = model.predict(valid_x) if valid_x is not None else None
        prob_test_x = model.predict(test_x) if test_x is not None else None
    else:
        prob_train_x = model.predict_proba(train_x)[:, 1] if train_x is not None else None
        prob_valid_x = model.predict_proba(valid_x)[:, 1] if valid_x is not None else None
        prob_test_x = model.predict_proba(test_x)[:, 1] if test_x is not None else None

    scores = {'pr_train': pr_auc_score(train_y, prob_train_x) if prob_train_x is not None else 0,
              'pr_valid': pr_auc_score(valid_y, prob_valid_x) if prob_valid_x is not None else 0,
              'pr_test': pr_auc_score(test_y, prob_test_x) if prob_test_x is not None else 0,
              'auc_train': roc_auc_score(train_y, prob_train_x) if prob_train_x is not None else 0,
              'auc_valid': roc_auc_score(valid_y, prob_valid_x) if prob_valid_x is not None else 0,
              'auc_test': roc_auc_score(test_y, prob_test_x) if prob_test_x is not None else 0}
    return scores


def enable_ebm_logging():
    # Enable Logging of EBM framework.
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)


def reduce_features(feature_columns, data):
    removed_feature_columns = []
    for feature_column in feature_columns:
        if ('(min' in feature_column or '(max' in feature_column or '(trend' in feature_column) or \
                (('per icu stay)' in feature_column) and ('(median' in feature_column or '(iqr' in feature_column)):
            removed_feature_columns.append(feature_column)
            data.drop(feature_column, axis=1, inplace=True)
    feature_columns = [f for f in feature_columns if f not in removed_feature_columns]
    assert len(feature_columns) == len(data.columns.tolist()) - 2
    print(f"Removed {len(removed_feature_columns)} features.")
    return feature_columns, data


def z_normalize_based_on_train(train_x, valid_x, test_x):
    normalization = StandardScaler()
    normalization.fit(train_x)
    train_x = pd.DataFrame(normalization.transform(train_x), index=train_x.index, columns=train_x.columns)
    valid_x = pd.DataFrame(normalization.transform(valid_x), index=valid_x.index, columns=valid_x.columns)
    test_x = pd.DataFrame(normalization.transform(test_x), index=test_x.index, columns=test_x.columns)
    return train_x, valid_x, test_x


if __name__ == '__main__':
    main()
