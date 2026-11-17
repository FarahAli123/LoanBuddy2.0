import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from src.prediction_service import load_prediction_artifacts, predict_loan


def test_load_prediction_artifacts():
    model, preprocessor = load_prediction_artifacts()

    assert isinstance(model, RandomForestClassifier)
    assert isinstance(preprocessor, ColumnTransformer)


def test_predict_sample_loan():
    model, preprocessor = load_prediction_artifacts()

    sample_loan = {}
    for _, pipeline, columns in preprocessor.transformers_:
        if columns:
            values = pipeline.named_steps["imputer"].statistics_
            sample_loan.update(zip(columns, values))

    predicted_class, default_score = predict_loan(
        sample_loan, model, preprocessor
    )

    assert predicted_class == 0
    assert default_score == pytest.approx(0.441, abs=0.001)