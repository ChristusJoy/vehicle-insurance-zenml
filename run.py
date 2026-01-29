from pipeline.training_pipeline import training_pipeline
from steps.data_ingestion import DataIngestionParameters
from steps.data_splitter import DataSplitterParameters
from steps.data_transformation import DataTransformationParameters
from steps.model_trainer import ModelTrainerParameters
from constant import COLLECTION_NAME, TARGET_COLUMN

if __name__ == "__main__":
    ingestion_params = DataIngestionParameters(collection_name=COLLECTION_NAME)
    splitter_params = DataSplitterParameters(target_column= TARGET_COLUMN)
    transformation_params = DataTransformationParameters()
    trainer_params = ModelTrainerParameters()

    training_pipeline(
        ingestion_params=ingestion_params,
        splitter_params=splitter_params,
        transformation_params=transformation_params,
        trainer_params=trainer_params,
        
    )
    
