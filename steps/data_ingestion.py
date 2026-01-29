import pandas as pd
from typing_extensions import Annotated
from pydantic import BaseModel
from utils.db_utils import MongoDBClient
from zenml import step, ArtifactConfig
from zenml.logger import get_logger

from constant import COLLECTION_NAME

logger = get_logger(__name__)


class DataIngestionParameters(BaseModel):
    """Parameters for MongoDB data ingestion."""
    collection_name: str = COLLECTION_NAME


@step
def ingest_data(
    params: DataIngestionParameters,
) -> Annotated[pd.DataFrame, ArtifactConfig(
        name="vehicle_insurance_raw_data",
        tags=["raw", "mongodb", "ingestion"],
    )]:
    from utils.db_utils import get_data_as_dataframe

    logger.info("Ingesting data from MongoDB")

    df = get_data_as_dataframe(
        collection_name=params.collection_name
    )

    logger.info(f"Shape of dataframe: {df.shape}")
    return df