from zenml import pipeline
from steps.data_ingestion import ingest_data, DataIngestionParameters
from steps.monitoring.detect_data_drift import detect_data_drift
from steps.monitoring.decide_retrain import decide_retrain


@pipeline
def monitoring_pipeline(
    collection_name: str,
    incoming_batch_tag: str,
):
    """
    Monitoring pipeline that detects data drift
    and decides whether retraining is needed.
    """

    # Load reference (historical) data
    reference_df = ingest_data(
        params=DataIngestionParameters(
            collection_name=collection_name,
            batch_tag="train",
        )
    )

    # Load incoming (new) data
    incoming_df = ingest_data(
        params=DataIngestionParameters(
            collection_name=collection_name,
            batch_tag=incoming_batch_tag,
        )
    )

    # Detect drift
    drift_detected, drifted_features = detect_data_drift(
        reference_df=reference_df,
        incoming_df=incoming_df,
    )

    # Decide whether to retrain
    retrain_required = decide_retrain(
        drift_detected=drift_detected
    )

    return drift_detected, drifted_features, retrain_required
