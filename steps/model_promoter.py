from zenml import step
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)

@step(enable_cache=False)
def promote_model():
    """
    Promotes the latest 'vehicle_insurance_model' version to 'current'.
    """
    client = Client()
    
    try:
        # Fetch the latest version of the model
        # The training step in the pipeline will have created a new version which is now 'latest'
        latest_version = client.get_model_version("vehicle_insurance_model", "latest")
        
        logger.info(f"Promoting model version '{latest_version.name}' to 'production' stage.")
        
        # Promote to 'production' stage (force=True to move the stage from any previous version)
        latest_version.set_stage("production", force=True)
        
    except Exception as e:
        logger.error(f"Failed to promote model: {e}")
        raise e
