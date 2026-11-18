# Description: Load the FRED economic data used by LoanBuddy.

from pathlib import Path

import pandas as pd


# Locate the raw data folder relative to this file.
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


class FredData:
    """Store the three FRED datasets together."""

    def __init__(self):
        """Load the unemployment rate, CPI, and federal funds rate."""
        self.unemployment_df = pd.read_csv(DATA_DIR / "UNRATE.csv")
        self.cpi_df = pd.read_csv(DATA_DIR / "CPIAUCSL.csv")
        self.interest_df = pd.read_csv(DATA_DIR / "FEDFUNDS.csv")

    def monthly_indicators(self):
        """Combine the FRED series by month and calculate annual inflation."""
        # Keep every date that appears in any of the three datasets.
        fred = self.unemployment_df.merge(
            self.cpi_df, on="observation_date", how="outer", validate="one_to_one"
        )
        fred = fred.merge(
            self.interest_df, on="observation_date", how="outer", validate="one_to_one"
        )

        # Sort the dates and keep one row per calendar month.
        fred["observation_date"] = pd.to_datetime(fred["observation_date"])
        fred = fred.set_index("observation_date").sort_index().asfreq("MS")

        # Compare each month's CPI with the same month a year earlier.
        last_year_cpi = fred["CPIAUCSL"].shift(12)
        fred["inflation_yoy_pct"] = (fred["CPIAUCSL"] / last_year_cpi - 1) * 100

        return fred