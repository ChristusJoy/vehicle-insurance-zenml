# Defines the ZenML training pipeline connecting all steps.
# Orchestrates data ingestion, splitting, transformation, training, and evaluation.

"""Defines the training pipeline: ingest, split, transform, train, eval."""

from zenml import pipeline
from zenml.logger import get_logger

from steps.data_ingestion import ingest_data, DataIngestionParameters
from steps.data_splitter import split_data, DataSplitterParameters
from steps.data_transformation import data_transformation, DataTransformationParameters
from steps.model_trainer import model_trainer, ModelTrainerParameters
from steps.model_evaluation import model_evaluation

logger = get_logger(__name__)


@pipeline(tags=["training", "vehicle_insurance"])
def training_pipeline(
    ingestion_params: DataIngestionParameters,
    splitter_params: DataSplitterParameters,
    transformation_params: DataTransformationParameters,
    trainer_params: ModelTrainerParameters,
):
    """
    Training pipeline.
    """
    raw_data = ingest_data(params=ingestion_params)
    X_train, X_test, y_train, y_test = split_data(df=raw_data, params=splitter_params)
    X_train_transformed, X_test_transformed, y_train_transformed, y_test_transformed, preprocessor = data_transformation(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        params=transformation_params
    )
    model, trained_preprocessor = model_trainer(
        X_train=X_train_transformed,
        y_train=y_train_transformed,
        preprocessor=preprocessor,  
        params=trainer_params
    )
    
    model_evaluation(
        model=model,
        X_test=X_test_transformed,
        y_test=y_test_transformed
    )