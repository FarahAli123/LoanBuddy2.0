from pathlib import Path

import joblib

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def load_prediction_artifacts():
    """Load the trained classifier and its fitted preprocessor."""
    model = joblib.load(MODELS_DIR / "random_forest_model.pkl")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    return model, preprocessor


if __name__ == "__main__":
    model, preprocessor = load_prediction_artifacts()
    print("Model loaded:", type(model).__name__)
    print("Preprocessor loaded:", type(preprocessor).__name__)