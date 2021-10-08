"""
Helper function for input/output operations.
"""

import pandas as pd
import numpy as np


def read_item_processing_descriptions_from_excel(item_overview_path):
    item_processing =\
        pd.read_excel(item_overview_path, sheet_name='item_processing', keep_default_na=False, usecols='A:C,T,AA:AP',
                      dtype={'QS_ID': np.int})
    # Rename QS_ID to id to merge item and merged item descriptions to process them together.
    item_processing = item_processing.rename(columns={'QS_ID': 'id'})
    merged_item_processing =\
        pd.read_excel(item_overview_path, sheet_name='merged_item_processing', keep_default_na=False, usecols='A:G',
                      converters={'id': lambda x: int(x) if str(x).isnumeric() else 0})
    merged_item_processing.drop(merged_item_processing[merged_item_processing['id'] < 10000].index, inplace=True)
    item_processing = item_processing.append(merged_item_processing, ignore_index=True)

    return item_processing
