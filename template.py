# Project scaffolding script to create the directory structure and initial files.
# Inputs: List of file paths to create. Outputs: Created directories and empty files.
# Iterates through a predefined list of files/folders and creates them if they don't exist.
import os
from pathlib import Path

project_name = "src"

list_of_files = [
    f"./steps/data_ingestion.py",  
    f"./steps/data_splitter.py",
    f"./steps/data_validation.py",
    f"./steps/data_transformation.py",
    f"./steps/model_trainer.py",
    f"./steps/model_evaluation.py",
    f"./configs/schema.yaml",
    f"./pipeline/training_pipeline.py",
    f"./pipeline/prediction_pipeline.py",
    f"./utils/main_utils.py",
    "run.py",
    "requirements.txt",
    ".env"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"file is already present at: {filepath}")