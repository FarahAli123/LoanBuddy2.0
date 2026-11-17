from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def load_prediction_artifacts():
    """Load the trained classifier and its fitted preprocessor."""
    model = joblib.load(MODELS_DIR / "random_forest_model.pkl")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    return model, preprocessor

def predict_loan(loan_details, model, preprocessor):
    """Return the predicted class and default score for one loan."""
    loan = pd.DataFrame([loan_details])
    processed_loan = preprocessor.transform(loan)

    predicted_class = int(model.predict(processed_loan)[0])
    default_index = model.classes_.tolist().index(1)
    default_score = float(model.predict_proba(processed_loan)[0, default_index])

    return predicted_class, default_score

if __name__ == "__main__":
    model, preprocessor = load_prediction_artifacts()
    print("Model loaded:", type(model).__name__)
    print("Preprocessor loaded:", type(preprocessor).__name__)