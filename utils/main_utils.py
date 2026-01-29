# Utility functions for general data processing tasks.
# Includes helper functions for mapping columns, creating dummies, etc.

"""General data utilities: mapping, dummy creation, renaming, and ID drop."""

import pandas as pd
from typing import List, Union
import logging

def map_gender_column(df: pd.DataFrame) -> pd.DataFrame:
    """Map Gender column to 0 for Female and 1 for Male."""
    if 'Gender' in df.columns:
        logging.info("Mapping 'Gender' column to binary values")
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
    return df

def create_dummy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create dummy variables for categorical features."""
    logging.info("Creating dummy variables for categorical features")
    df = pd.get_dummies(df, drop_first=True)
    return df

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename specific columns and ensure integer types for dummy columns."""
    logging.info("Renaming specific columns")
    renames = {
        "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
        "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
    }
    df = df.rename(columns=renames)
    
    # Cast boolean/dummy columns to int if they exist
    cols_to_convert = ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = df[col].astype('int')
    return df

def drop_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the 'id' column if it exists."""
    if 'id' in df.columns:
        logging.info("Dropping 'id' column")
        df = df.drop('id', axis=1)
    return df
