from pipelines.training_pipeline import training_pipeline
from pipelines.monitoring_pipeline import monitoring_pipeline
from zenml.client import Client

from steps.data_ingestion import DataIngestionParameters
from steps.data_splitter import DataSplitterParameters
from steps.data_transformation import DataTransformationParameters
from steps.model_trainer import ModelTrainerParameters

from constant import COLLECTION_NAME, TARGET_COLUMN


def run_training():
    training_pipeline(
        ingestion_params=DataIngestionParameters(
            collection_name=COLLECTION_NAME,
            batch_tag="train",
        ),
        splitter_params=DataSplitterParameters(
            target_column=TARGET_COLUMN,
        ),
        transformation_params=DataTransformationParameters(),
        trainer_params=ModelTrainerParameters(),
    )


def monitoring_and_retrain(batch_tag: str):
    # Run monitoring
    pipeline_run = monitoring_pipeline(
        collection_name=COLLECTION_NAME,
        incoming_batch_tag=batch_tag,
    )

    # Extract decision
    client = Client()
    run = client.get_pipeline_run(pipeline_run.id)
    retrain_required = run.steps["decide_retrain"].output.load()

    if retrain_required:
        print("Retraining triggered automatically.")
        run_training()
    else:
        print("No retraining needed.")


if __name__ == "__main__":
    monitoring_and_retrain("batch_2_drifted")
