import pandas as pd
from typing import List, Tuple
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

NUMERIC_FEATURES = ["Age", "Annual_Premium", "Vintage"]

MEAN_THRESHOLD = 0.20
IQR_THRESHOLD = 0.25
MIN_DRIFTED_FEATURES = 2


def _iqr(series: pd.Series) -> float:
    return series.quantile(0.75) - series.quantile(0.25)


@step
def detect_data_drift(
    reference_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
) -> Tuple[bool, List[str]]:
    """
    Detects data drift using mean and IQR comparison.
    """

    drifted_features = []

    logger.info("Starting data drift detection...")

    for feature in NUMERIC_FEATURES:
        ref_mean = reference_df[feature].mean()
        new_mean = incoming_df[feature].mean()

        mean_drift = abs(new_mean - ref_mean) / ref_mean
        iqr_drift = abs(
            _iqr(incoming_df[feature]) - _iqr(reference_df[feature])
        ) / _iqr(reference_df[feature])

        logger.info(
            f"{feature} | mean_drift={mean_drift:.3f}, iqr_drift={iqr_drift:.3f}"
        )

        if mean_drift > MEAN_THRESHOLD or iqr_drift > IQR_THRESHOLD:
            drifted_features.append(feature)

    drift_detected = len(drifted_features) >= MIN_DRIFTED_FEATURES

    logger.info(
        f"Drift detected={drift_detected} | Drifted features={drifted_features}"
    )

    return drift_detected, drifted_features
