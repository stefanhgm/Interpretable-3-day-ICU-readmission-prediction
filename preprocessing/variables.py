# Definition of all station in the data (these are either icus or imc stations)
station_mapping = {'10A OST': '10A OST', '10A_OST': '10A OST',
                   '10A WEST INT': '10A WEST', '10A_WEST_INT': '10A WEST',
                   '10B OST': '10B OST', '10B_OST': '10B OST',
                   '11A WEST C': '11A WEST C', '11A_WEST_C': '11A WEST C',
                   '11A WEST SU': '11A WEST SU', '11A_WEST_SU': '11A WEST SU',
                   '15B OST': '15B OST', '15B_OST': '15B OST', '15B_OST_INT': '15B OST',
                   '19A OST': '19A OST', '19A_OST': '19A OST',
                   '19B OST': '19B OST', '19B_OST': '19B OST',
                   'ANAES INT 2': 'ANAES INT 2', 'ANAES_INT_2': 'ANAES INT 2',
                   'ANAES INT 3': 'ANAES PAS', 'ANAES_PAS': 'ANAES PAS',
                   'CHIRURGIE 8': 'CHIRURGIE 8', 'CHIRURGIE_8': 'CHIRURGIE 8', 'CHIRURGIE_8_INT': 'CHIRURGIE 8',
                   'CHIRURGIE 9': 'CHIRURGIE 9', 'CHIRURGIE_9': 'CHIRURGIE 9'}
stations = ['10A OST', '10A WEST', '10B OST', '11A WEST C', '11A WEST SU', '15B OST', '19A OST', '19B OST',
            'ANAES INT 2', 'ANAES PAS', 'CHIRURGIE 8', 'CHIRURGIE 9']
im_icus = ['11A WEST C', '15B OST']
icus = stations
# We only consider cases from these four ICUs managed by ANIT-UKM.
included_stations = ['19A OST', '19B OST', 'ANAES INT 2', 'ANAES PAS']
anesthesiology_stations = included_stations + ['15B OST']

# Designated features for SAPS models.
saps_features = [
    'SAPS II Age (last)',
    'SAPS II Heart rate (min 1d)',
    'SAPS II Heart rate (max 1d)',
    'SAPS II Systolic blood pressure (min 1d)',
    'SAPS II Systolic blood pressure (max 1d)',
    'SAPS II Body core temperature (max 1d)',
    'SAPS II pO2/FiO2 (min 1d)',
    'SAPS II Urine (extrapolate 1d)',
    'SAPS II Blood Urea Nitrogen (max 1d)',
    'SAPS II Leucocytes (min 1d)',
    'SAPS II Leucocytes (max 1d)',
    'SAPS II Potassium (min 1d)',
    'SAPS II Potassium (max 1d)',
   'SAPS II Sodium (min 1d)',
    'SAPS II Sodium (max 1d)',
    'SAPS II Bicarbonate (min 1d)',
    'SAPS II Bilirubin total (max 1d)',
    'SAPS II GCS Score (min 1d)',
    'SAPS II Type of admission (SAPS II) (last per stay)',
    'SAPS II Metastatic cancer (SAPS II)',
    'SAPS II Hematologic malignancy (SAPS II)',
    'SAPS II AIDS (SAPS II)'
]
