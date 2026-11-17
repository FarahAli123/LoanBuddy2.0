# LoanBuddy 2.0

**Authors:** Janki Desai and Farah Ali

**Date:** November 17, 2026

**Description:** For our CPE 551 milestone, we built a desktop app that estimates default risk for a sample SBA loan. We wanted to give users a simple way to ask, “What would the model predict if this same loan had a different interest rate?” We also made a separate notebook to combine monthly economic data from FRED.

## How to run the app

We tested LoanBuddy with Python 3.14.7. Open the `LoanBuddy2.0` project in PyCharm and run these commands in the **Terminal** tab from the folder that contains this README.

First, check that Python can open a Tkinter window:

```bash
python -m tkinter
```

If a small test window opens, close it and continue. If the terminal says `python` is unavailable, try `python3` and use it for the rest of the commands too. If Python says `tkinter` or `_tkinter` is missing, use a Python installation that includes Tkinter. The standard installer from python.org includes it on Mac and Windows.

Install the project dependencies:

```bash
python -m pip install scikit-learn==1.9.1 numpy==2.5.4 scipy==1.18.1 pandas==3.1.1 joblib==1.6.0
```

Start LoanBuddy:

```bash
python -m gui.loanbuddy_gui
```

The app loads `random_forest_model.pkl` and `preprocessor.pkl` from the `models/` folder, so you can run it without retraining the model.

## Try a loan scenario

1. Click **Predict** with the prefilled interest rate. Look for the predicted class and default score below the button.
2. Enter `8.5` for an interest rate of 8.5% and click **Predict** again.
3. Try `12` for 12% and compare the scores. The other details of the sample loan stay the same, so you’re only changing one input. The predicted class may stay the same even if the score changes.

## How to run the tests

```bash
python -m pip install pytest
python -m pytest tests/test_prediction_service.py -q
```

The two tests check that the saved model and preprocessor load and that the sample loan gives the expected prediction.

## About the notebooks

- `01_sba_exploration.ipynb` shows how we prepared the SBA loan data and trained the model. You’ll need the original SBA CSVs to rerun it.
- `02_fred_exploration.ipynb` reads `UNRATE.csv`, `CPIAUCSL.csv`, and `FEDFUNDS.csv` from `data/raw/`. Run all cells in order to combine the monthly data, calculate year-over-year inflation where the required CPI values are available, and save `data/processed/fred_monthly.csv`. The prediction app runs on its own.