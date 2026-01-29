from zenml import pipeline
from zenml.logger import get_logger

from steps.data_ingestion import ingest_data, DataIngestionParameters
from steps.data_splitter import split_data, DataSplitterParameters

logger = get_logger(__name__)


@pipeline(tags=["training", "vehicle_insurance"])
def training_pipeline(
    ingestion_params: DataIngestionParameters,
    splitter_params: DataSplitterParameters,
):
    """
    Training pipeline.
    """
    raw_data = ingest_data(params=ingestion_params)
    X_train, X_test, y_train, y_test = split_data(df=raw_data, params=splitter_params)
    return X_train, X_test, y_train, y_test