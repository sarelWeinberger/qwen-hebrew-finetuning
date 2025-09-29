import os
import pandas as pd
from src.translate_func import dict_to_prompt


def add_dataset_to_csv(file_name, columns_name, dataset, sample_to_dict_func):
    # Create a new .csv file if not already exists
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        df = pd.DataFrame()
    # Add the new columns
    df[columns_name] = [dict_to_prompt(sample_to_dict_func(exmp)) for exmp in dataset]
    # Save to csv
    df.to_csv(file_name, index=False)

    return df