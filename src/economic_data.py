from pathlib import Path

import pandas as pd


def load_fred_data(raw_dir):
    """Read the three FRED files from the raw data folder."""
    folder = Path(raw_dir)

    unemployment = pd.read_csv(folder / "UNRATE.csv")
    cpi = pd.read_csv(folder / "CPIAUCSL.csv")
    federal_funds = pd.read_csv(folder / "FEDFUNDS.csv")

    return unemployment, cpi, federal_funds