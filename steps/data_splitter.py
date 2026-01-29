from typing import Tuple
import pandas as pd
from typing_extensions import Annotated
from zenml import step, ArtifactConfig
from zenml.logger import get_logger
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)

class DataSplitterParameters(BaseModel):
    """Parameters for data splitting."""
    test_size: float = 0.2
    random_state: int = 42
    target_column: str = "Response"

@step
def split_data(
    df: pd.DataFrame,
    params: DataSplitterParameters,
) -> Tuple[
    Annotated[pd.DataFrame, ArtifactConfig(name="X_train", tags=["train", "features"])],
    Annotated[pd.DataFrame, ArtifactConfig(name="X_test", tags=["test", "features"])],
    Annotated[pd.Series, ArtifactConfig(name="y_train", tags=["train", "labels"])],
    Annotated[pd.Series, ArtifactConfig(name="y_test", tags=["test", "labels"])],
]:
    """
    Splits the data into training and testing sets.
    """
    logger.info("Splitting data into train and test sets")
    
    if params.target_column not in df.columns:
        raise ValueError(f"Target column '{params.target_column}' not found in dataframe. Available columns: {df.columns.tolist()}")

    X = df.drop(columns=[params.target_column])
    y = df[params.target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params.test_size, random_state=params.random_state
    )

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test
