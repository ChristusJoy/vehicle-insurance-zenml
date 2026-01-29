# Vehicle Insurance Prediction with ZenML

An end-to-end Machine Learning Operations (MLOps) project for vehicle insurance prediction, built using [ZenML](https://zenml.io/), MongoDB, and Scikit-learn.

## Project Overview

This project implements a complete training pipeline that ingests data from a MongoDB database, processes it, trains a Random Forest Classifier (handling class imbalance with SMOTEENN), and evaluates the model performance.

## Detailed Pipeline Breakdown

Each step in the pipeline is designed to be modular and reusable.

### 1. Data Ingestion
- **Source**: MongoDB Collection (`vehicle-insurance-data`).
- **Function**: Reads data efficiently using pandas and PyMongo.
- **Output**: Raw pandas DataFrame.

### 2. Data Splitting
- **Strategy**: classic Train-Test Split.
- **Ratio**: 80% Training, 20% Testing (configurable).
- **Random State**: Fixed for reproducibility (default: 42).

### 3. Data Transformation
A comprehensive preprocessing step that prepares data for ML:
- **Feature Engineering**:
    - `Gender`: Mapped to binary (Male: 1, Female: 0).
    - `id`: Unique identifier column is dropped.
- **Scaling & Encoding** (`ColumnTransformer`):
    - **StandardScaler**: Applied to `Age`, `Annual_Premium`, `Vintage`.
    - **MinMaxScaler**: Applied to `Policy_Sales_Channel`.
    - **OneHotEncoder**: Applied to categorical features `Vehicle_Age` and `Vehicle_Damage`.
- **Output**: Transformed Numpy arrays for X and y, and the fitted Scikit-learn Pipeline object.

### 4. Model Training
Handles class imbalance and trains the robust Random Forest model:
- **Imbalance Handling**: Uses **SMOTEENN** (Combine over-sampling using SMOTE and under-sampling using Edited Nearest Neighbours) to address potential data skew.
- **Algorithm**: **RandomForestClassifier**.
- **Parameters**: `n_estimators=350`, `random_state=42`.
- **Output**: Trained Scikit-learn classifier model artifact.

### 5. Model Evaluation
Assess model performance on unseen test data:
- **Metrics Calculated**:
    - **ROC AUC Score**: A core metric for binary classification.
    - **Precision & Recall**: Essential for checking how well the model identifies positive cases (interested customers).
    - **Confusion Matrix**: For detailed error analysis.
- **Logging**: All metrics are logged as artifacts in ZenML.

## Dataset Information

The project expects a dataset in MongoDB with the following schema characteristics:
- **Target**: `Response` (Binary: 1 for Interested, 0 for Not Interested).
- **Categorical Features**: `Gender`, `Vehicle_Age`, `Vehicle_Damage`.
- **Numerical Features**: `Age`, `Annual_Premium`, `Vintage`, `Policy_Sales_Channel`.

## Prerequisites

- Python 3.8 or higher
- MongoDB installed and running (locally or accessible remotely)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd vehicle-insurance-zenml
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize ZenML:**
   If you haven't used ZenML before, initialize it:
   ```bash
   zenml init
   ```

## Configuration

This project uses environment variables for configuration, specifically for the database connection.

1. Create a `.env` file in the root directory of the project.
2. Add the following environment variables:

   ```env
   MONGODB_URL="mongodb://localhost:27017"
   MONGO_DB_NAME="vehicle-insurance"
   ```

   *   `MONGODB_URL`: Your MongoDB connection string.
   *   `MONGO_DB_NAME`: (Optional) The name of the database to use. Defaults to `vehicle-insurance` if not set.

> **Note:** Ensure your MongoDB instance contains the required data in the `vehicle-insurance-data` collection (defined in `constant.py`), or update the configuration accordingly.

## Running the Project

To execute the training pipeline, run the `run.py` script:

```bash
python run.py
```

This script initializes the pipeline steps and executes the training workflow.

## Viewing Results

You can view the run artifacts and pipeline status using the ZenML dashboard:

```bash
zenml login --local
```

This will ensure the local ZenML server is running, and you can access the dashboard in your browser (usually at `http://127.0.0.1:8237`).

## Project Structure

*   `pipeline/`: Contains the ZenML pipeline definition.
*   `steps/`: Individual steps for ingestion, splitting, transformation, training, and evaluation.
*   `utils/`: Helper functions for database connection and data processing.
*   `run.py`: Entry point script to run the pipeline.
*   `constant.py`: Global constants.