# ZenML step for training the machine learning model.
# Trains a Random Forest classifier with SMOTEENN for class imbalance.

"""Balances classes with SMOTEENN and trains a RandomForest classifier."""

from typing import Tuple
import numpy as np
from zenml import step, ArtifactConfig, Model
from zenml.logger import get_logger
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
from typing_extensions import Annotated
from pydantic import BaseModel

logger = get_logger(__name__)

class ModelTrainerParameters(BaseModel):
    """Parameters for model training."""
    n_estimators: int = 350
    random_state: int = 42

@step(
    model=Model(
        name="vehicle_insurance_model",  
        description="RandomForest model for vehicle insurance claim prediction",
    )
)

def model_trainer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    preprocessor,   
    params: ModelTrainerParameters,
) -> Tuple[
    Annotated[ClassifierMixin, ArtifactConfig(name="model", tags=["model"])],
    Annotated[object, ArtifactConfig(name="preprocessor", tags=["preprocessing"])],
]:
    """
    Model training step that handles:
    1. SMOTEENN for class imbalance
    2. Model training (RandomForest)
    """
    try:
        logger.info("Starting Model Training...")
        
        # 1. Apply SMOTEENN
        logger.info("Applying SMOTEENN to training data...")
        smt = SMOTEENN(sampling_strategy="minority", random_state=params.random_state)
        X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train)
        logger.info(f"SMOTEENN complete. Resampled shape: {X_train_resampled.shape}")
        
        # 2. Train Model
        logger.info(f"Training RandomForestClassifier with n_estimators={params.n_estimators}...")
        model = RandomForestClassifier(
            n_estimators=params.n_estimators, 
            random_state=params.random_state
        )
        model.fit(X_train_resampled, y_train_resampled)
        logger.info("Model training complete.")
        
        return model, preprocessor
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise e
