import numpy as np
from typing import Tuple
from typing_extensions import Annotated
from zenml import step, ArtifactConfig
from zenml.logger import get_logger
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix

logger = get_logger(__name__)

@step
def model_evaluation(
    model: ClassifierMixin,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[
    Annotated[float, ArtifactConfig(name="roc_auc_score", tags=["metric"])],
    Annotated[float, ArtifactConfig(name="precision_score", tags=["metric"])],
    Annotated[float, ArtifactConfig(name="recall_score", tags=["metric"])],
    Annotated[np.ndarray, ArtifactConfig(name="confusion_matrix", tags=["metric"])],
]:
    """
    Model evaluation step.

    Args:
        model: Trained classifier model.
        X_test: Transformed testing features.
        y_test: Testing labels.

    Returns:
        Tuple containing ROC-AUC, Precision, Recall, and Confusion Matrix.
    """
    try:
        logger.info("Starting Model Evaluation...")

        # Predict probabilities for ROC-AUC
        # RandomForest (used in trainer) has predict_proba
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            logger.warning("Model does not support predict_proba, using predict for ROC-AUC.")
            y_pred_proba = model.predict(X_test)
        
        y_pred = model.predict(X_test)

        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)

        logger.info(f"ROC-AUC: {roc_auc:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")

        return roc_auc, precision, recall, conf_matrix

    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise e
