# Description: Load the saved LoanBuddy model and predict loan default.

from pathlib import Path

import joblib
import pandas as pd


# Locate the models folder relative to this file.
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def load_prediction_artifacts():
    """Load the trained classifier and its fitted preprocessor."""
    model = joblib.load(MODELS_DIR / "random_forest_model.pkl")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    return model, preprocessor


def predict_loan(loan_details, model, preprocessor):
    """Return the predicted class and default score for one loan."""
    # Convert the loan details into a one-row table and prepare it for the model.
    loan = pd.DataFrame([loan_details])
    processed_loan = preprocessor.transform(loan)

    # Get the predicted class for the single loan.
    predicted_class = int(model.predict(processed_loan)[0])

    # Find the score column for class 1, which represents default.
    default_index = model.classes_.tolist().index(1)
    default_score = float(
        model.predict_proba(processed_loan)[0, default_index]
    )

    return predicted_class, default_score


class LoanPredictor:
    """Keep the model and preprocessor together for repeated predictions."""

    def __init__(self):
        """Load the model and preprocessor for this predictor."""
        self.model, self.preprocessor = load_prediction_artifacts()

    def predict(self, loan_details):
        """Return the predicted class and default score for one loan."""
        return predict_loan(loan_details, self.model, self.preprocessor)


if __name__ == "__main__":
    # Check that creating a predictor loads both saved files.
    predictor = LoanPredictor()
    print("Model loaded:", type(predictor.model).__name__)
    print("Preprocessor loaded:", type(predictor.preprocessor).__name__)