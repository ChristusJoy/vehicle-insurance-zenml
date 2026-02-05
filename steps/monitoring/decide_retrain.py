from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def decide_retrain(drift_detected: bool) -> bool:
    """
    Decide whether retraining is required.
    """
    if drift_detected:
        logger.warning("Data drift detected — retraining required.")
        return True

    logger.info("No significant drift detected — retraining not required.")
    return False
