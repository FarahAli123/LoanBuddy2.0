import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from src.prediction_service import (
    LoanPredictor,
    load_prediction_artifacts,
    predict_loan,
)

def _build_sample_loan(preprocessor):
    """Build the sample loan from the fitted imputer values."""
    sample_loan = {}
    for _, pipeline, columns in preprocessor.transformers_:
        if columns:
            values = pipeline.named_steps["imputer"].statistics_
            sample_loan.update(zip(columns, values))
    return sample_loan

def test_load_prediction_artifacts():
    """Check that the saved model and preprocessor load."""
    model, preprocessor = load_prediction_artifacts()

    assert isinstance(model, RandomForestClassifier)
    assert isinstance(preprocessor, ColumnTransformer)

def test_predict_sample_loan():
    """Check the prediction returned by the service function."""
    model, preprocessor = load_prediction_artifacts()
    sample_loan = _build_sample_loan(preprocessor)

    predicted_class, default_score = predict_loan(
        sample_loan, model, preprocessor
    )

    assert predicted_class == 0
    assert default_score == pytest.approx(0.441, abs=0.001)

def test_loan_predictor_predicts_sample_loan():
    """Check the prediction method used by the app."""
    predictor = LoanPredictor()
    sample_loan = _build_sample_loan(predictor.preprocessor)

    predicted_class, default_score = predictor.predict(sample_loan)

    assert predicted_class == 0
    assert default_score == pytest.approx(0.441, abs=0.001)