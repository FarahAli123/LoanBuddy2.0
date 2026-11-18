# LoanBuddy 2.0

**Janki Desai and Farah Ali**  
CPE 551 Course Project Milestone · November 17, 2026

LoanBuddy is a desktop demo that uses a trained Random Forest model to estimate default risk for a sample SBA loan. You can change the sample loan’s initial interest rate and compare the model’s predictions. A separate notebook combines monthly economic data from FRED and calculates year-over-year inflation.

## Set up the project

We tested LoanBuddy with Python 3.14.7. Open the `LoanBuddy2.0` project in PyCharm and run the commands below in the **Terminal** tab from the folder that contains this README. If your terminal uses `python3`, substitute it for `python` in each command.

First, check that your Python installation can open a Tkinter window.

```bash
python -m tkinter
```

Close the test window once it opens. If Python reports that `tkinter` or `_tkinter` is missing, use a Python installation that includes Tkinter.

Install the dependencies used to run the prediction demo and process the data.

```bash
python -m pip install scikit-learn==1.9.1 numpy==2.5.4 scipy==1.18.1 pandas==3.1.1 joblib==1.6.0
```

## Run the prediction demo

```bash
python -m gui.loanbuddy_gui
```

The app uses `LoanPredictor` to load `models/random_forest_model.pkl` and `models/preprocessor.pkl` at startup. These saved files let the demo run without retraining the model or opening either notebook.

1. Click **Predict** with the prefilled interest rate. The app displays a predicted class and a model default score.
2. Enter `8.5` for an interest rate of 8.5% and click **Predict** again.
3. Try `12` for 12% and compare the scores. The predicted class may stay the same even if the score changes.

Only the interest-rate input changes across these scenarios. The score reflects the model’s prediction for the sample loan.

## Run the automated tests

```bash
python -m pip install pytest
python -m pytest tests/ -q
```

The full suite contains **five tests**. Three check that the saved model and preprocessor load and that the sample loan prediction works through both the service function and the `LoanPredictor` class. Two check that `FredData` combines the monthly series by date and calculates year-over-year inflation, including when the prior year’s CPI is missing.

## Work with the notebooks

### SBA loan exploration

`01_sba_exploration.ipynb` shows how we explored and cleaned the SBA loan records, prepared the model inputs, and compared prediction models. To rerun it from the beginning, place the original SBA 7(a) CSVs in `data/raw/` and run the cells in order.

### FRED economic data

`02_fred_exploration.ipynb` uses the `FredData` class in `src/fred_data.py`. To rerun it, make sure these three files are in `data/raw/`:

- `UNRATE.csv` — unemployment rate
- `CPIAUCSL.csv` — Consumer Price Index
- `FEDFUNDS.csv` — federal funds rate

Run the notebook cells in order. The notebook combines the series by month, calculates year-over-year inflation where both CPI values are available, and saves the result to `data/processed/fred_monthly.csv`. The prediction demo can run independently of this notebook.