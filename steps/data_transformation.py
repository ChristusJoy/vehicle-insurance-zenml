# ZenML step for data preprocessing and feature engineering.
# Handles scaling, encoding, and other data transformations.

"""Applies pandas fixes and sklearn preprocessing to features and labels."""

import pandas as pd
import numpy as np
from typing import Tuple, List
from typing_extensions import Annotated
from zenml import step, ArtifactConfig
from zenml.logger import get_logger
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from utils.main_utils import (
    map_gender_column,
    drop_id_column
)

logger = get_logger(__name__)

class DataTransformationParameters(BaseModel):
    """Parameters for data transformation."""
    num_features: List[str] = ["Age", "Annual_Premium", "Vintage"]
    mm_columns: List[str] = ["Policy_Sales_Channel"]
    cat_features: List[str] = ["Vehicle_Age", "Vehicle_Damage"]
    # Add other columns if needed, defaults are provided

@step
def data_transformation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: DataTransformationParameters,
) -> Tuple[
    Annotated[np.ndarray, ArtifactConfig(name="X_train_transformed", tags=["train", "transformed"])],
    Annotated[np.ndarray, ArtifactConfig(name="X_test_transformed", tags=["test", "transformed"])],
    Annotated[np.ndarray, ArtifactConfig(name="y_train_transformed", tags=["train", "labels"])],
    Annotated[np.ndarray, ArtifactConfig(name="y_test_transformed", tags=["test", "labels"])],
    Annotated[Pipeline, ArtifactConfig(name="preprocessor", tags=["preprocessor"])],
]:
    """
    Data transformation step that handles:
    1. Custom pandas-based preprocessing (Gender map, drop ID)
    2. Scikit-learn Pipeline (Scaling, OneHotEncoding)
    """
    try:
        logger.info("Starting Data Transformation...")

        # 1. Apply custom transformations
        # We apply the same pandas transformations to both train and test
        X_train = map_gender_column(X_train)
        X_train = drop_id_column(X_train)

        X_test = map_gender_column(X_test)
        X_test = drop_id_column(X_test)
        
        logger.info("Custom pandas transformations applied.")

        # 2. Create and Fit Preprocessor
        numeric_transformer = StandardScaler()
        min_max_scaler = MinMaxScaler()
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ("StandardScaler", numeric_transformer, params.num_features),
                ("MinMaxScaler", min_max_scaler, params.mm_columns),
                ("OneHotEncoder", one_hot_encoder, params.cat_features),
            ],
            remainder='passthrough'
        )

        pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])

        logger.info("Fitting preprocessor on training data...")
        X_train_arr = pipeline.fit_transform(X_train)
        X_test_arr = pipeline.transform(X_test)
        logger.info("Preprocessor fit and transform complete.")

        # Convert y to numpy arrays
        y_train_arr = np.array(y_train)
        y_test_arr = np.array(y_test)

        return (
            X_train_arr,
            X_test_arr,
            y_train_arr,
            y_test_arr,
            pipeline
        )

    except Exception as e:
        logger.error(f"Error in data transformation: {e}")
        raise e
