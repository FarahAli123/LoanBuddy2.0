import pandas as pd
import pytest

from src import fred_data


@pytest.fixture
def fred_with_sample_data(tmp_path, monkeypatch):
    """Load small FRED sample files from a temporary folder."""
    (tmp_path / "UNRATE.csv").write_text(
        "observation_date,UNRATE\n2020-01-01,4.0\n",
        encoding="utf-8",
    )
    (tmp_path / "CPIAUCSL.csv").write_text(
        "observation_date,CPIAUCSL\n"
        "2020-01-01,100.0\n"
        "2021-01-01,105.0\n"
        "2021-02-01,106.0\n",
        encoding="utf-8",
    )
    (tmp_path / "FEDFUNDS.csv").write_text(
        "observation_date,FEDFUNDS\n2020-01-01,1.5\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(fred_data, "DATA_DIR", tmp_path)
    return fred_data.FredData()


def test_monthly_indicators_combines_series_by_date(fred_with_sample_data):
    """Check that the three indicators appear on the same monthly row."""
    monthly = fred_with_sample_data.monthly_indicators()
    january = monthly.loc["2020-01-01"]

    assert january["UNRATE"] == 4.0
    assert january["CPIAUCSL"] == 100.0
    assert january["FEDFUNDS"] == 1.5


def test_monthly_indicators_uses_cpi_from_same_month_last_year(
    fred_with_sample_data,
):
    """Check inflation for months with and without prior-year CPI."""
    monthly = fred_with_sample_data.monthly_indicators()

    # January CPI rose from 100 to 105. February 2020 CPI is missing.
    assert monthly.loc["2021-01-01", "inflation_yoy_pct"] == pytest.approx(5.0)
    assert pd.isna(monthly.loc["2021-02-01", "inflation_yoy_pct"])