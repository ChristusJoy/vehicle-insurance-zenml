from pipeline.training_pipeline import training_pipeline
from steps.data_ingestion import DataIngestionParameters
from steps.data_splitter import DataSplitterParameters
from constant import COLLECTION_NAME

if __name__ == "__main__":
    ingestion_params = DataIngestionParameters(
        collection_name=COLLECTION_NAME
    )
    
    splitter_params = DataSplitterParameters(
        target_column="Response"
    )

    training_pipeline(
        ingestion_params=ingestion_params,
        splitter_params=splitter_params
    )