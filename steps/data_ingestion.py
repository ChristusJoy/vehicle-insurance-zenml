import pandas as pd
from typing import Optional
from typing_extensions import Annotated
from pydantic import BaseModel, Field

from zenml import step, ArtifactConfig
from zenml.logger import get_logger

from constant import COLLECTION_NAME
from utils.db_utils import get_data_as_dataframe

logger = get_logger(__name__)


class DataIngestionParameters(BaseModel):
    """
    Parameters for MongoDB data ingestion.
    """
    collection_name: str = Field(
        default=COLLECTION_NAME,
        description="MongoDB collection to read data from"
    )
    batch_tag: Optional[str] = Field(
        default=None,
        description="Optional batch tag to filter data (e.g. train, batch_1_clean, batch_2_drifted)"
    )


@step(enable_cache=False)
def ingest_data(
    params: DataIngestionParameters,
) -> Annotated[
    pd.DataFrame,
    ArtifactConfig(
        name="vehicle_insurance_raw_data",
        tags=["raw", "mongodb", "ingestion"],
    ),
]:

    logger.info("Starting data ingestion from MongoDB")

    query = {}
    if params.batch_tag is not None:
        query["batch_tag"] = params.batch_tag
        logger.info(f"Filtering data using batch_tag='{params.batch_tag}'")
    else:
        logger.info("No batch_tag provided — loading full collection")

    df = get_data_as_dataframe(
        collection_name=params.collection_name,
        query=query,
    )

    if df.empty:
        raise ValueError(
            f"No data found in collection '{params.collection_name}' "
            f"for batch_tag='{params.batch_tag}'"
        )

    logger.info(f"Data ingestion completed | Shape: {df.shape}")

    return df
