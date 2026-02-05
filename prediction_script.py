import pandas as pd
from zenml.client import Client


def load_current_model():
    """
    Load the CURRENT or LATEST model version along with its preprocessor
    from the ZenML Model Registry.
    """
    client = Client()

    try:
        version = client.get_model_version("vehicle_insurance_model", "current")
        print("Loaded CURRENT model version")
    except Exception:
        version = client.get_model_version("vehicle_insurance_model", "latest")
        print("No current model found, using LATEST version")

    model = version.get_artifact("model").load()
    preprocessor = version.get_artifact("preprocessor").load()

    return model, preprocessor


def main():
    model, preprocessor = load_current_model()

    # Raw user input ONLY (no manual feature hacks)
    sample = pd.DataFrame([{
  "Gender": "Male",
  "Age": 43,
  "Driving_License": 1,
  "Region_Code": 8,
  "Previously_Insured": 0,
  "Vehicle_Age": "1-2 Year",
  "Vehicle_Damage": "Yes",
  "Annual_Premium": 30115,
  "Policy_Sales_Channel": 124,
  "Vintage": 247,
}])

    # Apply the SAME preprocessing used during training
    X = preprocessor.transform(sample)

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    print("Prediction (0 = No Claim, 1 = Claim):", prediction)
    print("Claim Probability:", round(float(probability), 4))


if __name__ == "__main__":
    main()
