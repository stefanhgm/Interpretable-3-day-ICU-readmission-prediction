import numpy as np
import pandas as pd

seed = 87

# Assume empty values indicate a "normal" value, so assign 0 points. Use nan values to indicate errors.
# According to: A New Simplified Acute Physiology Score (SAPS II) Based on a European/North American Multicenter Study.
# (https://jamanetwork.com/journals/jama/article-abstract/409979)
saps_scoring_functions = {
    'Age': lambda row:
    0 if np.isnan(row['Age (static all data) [years]']) or row['Age (static all data) [years]'] < 40 else
    (7 if 40 <= row['Age (static all data) [years]'] < 60 else
     (12 if 60 <= row['Age (static all data) [years]'] < 70 else
      (15 if 70 <= row['Age (static all data) [years]'] < 75 else
       (16 if 75 <= row['Age (static all data) [years]'] < 80 else 18)))),
    'Heart rate': lambda row:
    0 if (np.isnan(row['Heart rate (min 1d) [bpm]']) or np.isnan(row['Heart rate (max 1d) [bpm]'])) or
         (row['Heart rate (min 1d) [bpm]'] >= 70 and row['Heart rate (max 1d) [bpm]'] < 120) else
    (11 if row['Heart rate (min 1d) [bpm]'] < 40 else
     (7 if row['Heart rate (max 1d) [bpm]'] >= 160 else
      (4 if row['Heart rate (max 1d) [bpm]'] >= 120 else
       (2 if row['Heart rate (min 1d) [bpm]'] < 70 else np.nan)))),
    'Systolic BP': lambda row:
    0 if (np.isnan(row['Systolic blood pressure (min 1d) [mmHg]']) or
          np.isnan(row['Systolic blood pressure (max 1d) [mmHg]'])) or
         (row['Systolic blood pressure (min 1d) [mmHg]'] >= 100 and
          row['Systolic blood pressure (max 1d) [mmHg]'] < 200) else
    (13 if row['Systolic blood pressure (min 1d) [mmHg]'] < 70 else
     (5 if row['Systolic blood pressure (min 1d) [mmHg]'] < 100 else
      (2 if row['Systolic blood pressure (max 1d) [mmHg]'] >= 200 else np.nan))),
    'Body temperature': lambda row:
    0 if np.isnan(row['Body core temperature (max 1d) [°C]']) or
         row['Body core temperature (max 1d) [°C]'] < 39 else
    (3 if row['Body core temperature (max 1d) [°C]'] >= 39 else np.nan),
    'PaO2/FiO2': lambda row:
    0 if np.isnan(row['paO2/FiO2 (min 1d) [mmHg/FiO2]']) else
    (5 if row['paO2/FiO2 (min 1d) [mmHg/FiO2]'] >= 200 else
     (9 if row['paO2/FiO2 (min 1d) [mmHg/FiO2]'] >= 100 else
      11 if row['paO2/FiO2 (min 1d) [mmHg/FiO2]'] < 100 else np.nan)),
    'Urinary output': lambda row:
    0 if np.isnan(row['Urine volume out (extrapolate 1d) [mL]']) or
         row['Urine volume out (extrapolate 1d) [mL]'] >= 1 else
    (4 if row['Urine volume out (extrapolate 1d) [mL]'] >= 0.5 else
     (11 if row['Urine volume out (extrapolate 1d) [mL]'] < 0.5 else np.nan)),
    'Serum urea level': lambda row:
    0 if np.isnan(row['Blood Urea Nitrogen (max 1d) [mg/dL]']) or row['Blood Urea Nitrogen (max 1d) [mg/dL]'] < 28 else
    (6 if row['Blood Urea Nitrogen (max 1d) [mg/dL]'] < 84 else
     10 if row['Blood Urea Nitrogen (max 1d) [mg/dL]'] >= 84 else np.nan),
    'WBC count': lambda row:
    0 if (np.isnan(row['Leucocytes (min 1d) [thousand/μL]']) or np.isnan(row['Leucocytes (max 1d) [thousand/μL]'])) or
         (row['Leucocytes (min 1d) [thousand/μL]'] >= 1000 and row['Leucocytes (max 1d) [thousand/μL]'] < 20000) else
    (3 if row['Leucocytes (max 1d) [thousand/μL]'] >= 20000 else
     (12 if row['Leucocytes (min 1d) [thousand/μL]'] < 1000 else np.nan)),
    'Serum potassium': lambda row:
    0 if (np.isnan(row['Potassium (min 1d) [mmol/L]']) or np.isnan(row['Potassium (max 1d) [mmol/L]'])) or
         (row['Potassium (min 1d) [mmol/L]'] >= 3 and row['Potassium (max 1d) [mmol/L]'] < 5) else
    (3 if row['Potassium (min 1d) [mmol/L]'] < 3 else
     (3 if row['Potassium (max 1d) [mmol/L]'] >= 5 else np.nan)),
    'Serum sodium': lambda row:
    0 if (np.isnan(row['Sodium (min 1d) [mmol/L]']) or np.isnan(row['Sodium (max 1d) [mmol/L]'])) or
         (row['Sodium (min 1d) [mmol/L]'] >= 125 and row['Sodium (max 1d) [mmol/L]'] < 145) else
    (5 if row['Sodium (min 1d) [mmol/L]'] < 125 else
     (1 if row['Sodium (max 1d) [mmol/L]'] >= 145 else np.nan)),
    'Serum bicarbonate': lambda row:
    0 if np.isnan(row['Bicarbonate (min 1d) [mmol/L]']) or row['Bicarbonate (min 1d) [mmol/L]'] >= 20 else
    (3 if row['Bicarbonate (min 1d) [mmol/L]'] >= 15 else
     (6 if row['Bicarbonate (min 1d) [mmol/L]'] < 15 else np.nan)),
    'Bilirubin level': lambda row:
    0 if np.isnan(row['Bilirubin total (max 1d) [mg/dL]']) or row['Bilirubin total (max 1d) [mg/dL]'] < 4 else
    (4 if row['Bilirubin total (max 1d) [mg/dL]'] < 6 else
     9 if row['Bilirubin total (max 1d) [mg/dL]'] >= 6 else np.nan),
    'Glasgow coma score': lambda row:
    0 if np.isnan(row['GCS score (min 1d)']) or row['GCS score (min 1d)'] >= 14 else
    (5 if row['GCS score (min 1d)'] >= 11 else
     (7 if row['GCS score (min 1d)'] >= 9 else
      (13 if row['GCS score (min 1d)'] >= 6 else
       26 if row['GCS score (min 1d)'] < 6 else np.nan))),
    'Type of admission': lambda row:
    0 if (pd.isna(row['Type of admission (static per hospital stay)']) or
          row['Type of admission (static per hospital stay)'] == 'nan' or
          row['Type of admission (static per hospital stay)'] == 'Scheduled surgical') else
    (6 if row['Type of admission (static per hospital stay)'] == 'Medical' else
     (8 if row['Type of admission (static per hospital stay)'] == 'Unscheduled surgical' else np.nan)),
    'Chronic disease metastatic cancer': lambda row:
    0 if (pd.isna(row['Metastatic cancer (static all data)']) or
          row['Metastatic cancer (static all data)'] == 'nan' or
          not row['Metastatic cancer (static all data)']) else
    (9 if row['Metastatic cancer (static all data)'] else np.nan),
    'Chronic disease hematologic malignancy': lambda row:
    0 if (pd.isna(row['Hematologic malignancy (static all data)']) or
          row['Hematologic malignancy (static all data)'] == 'nan' or
          not row['Hematologic malignancy (static all data)']) else
    (10 if row['Hematologic malignancy (static all data)'] else np.nan),
    'Chronic disease AIDS': lambda row:
    0 if (pd.isna(row['AIDS (static all data)']) or
          row['AIDS (static all data)'] == 'nan' or
          not row['AIDS (static all data)']) else
    (17 if row['AIDS (static all data)'] else np.nan)
}
