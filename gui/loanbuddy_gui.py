import math
import tkinter as tk
from tkinter import ttk

from src.prediction_service import load_prediction_artifacts, predict_loan


def main():
    model, preprocessor = load_prediction_artifacts()

    sample_loan = {}
    for _, pipeline, columns in preprocessor.transformers_:
        if columns:
            values = pipeline.named_steps["imputer"].statistics_
            sample_loan.update(zip(columns, values))

    window = tk.Tk()
    window.title("LoanBuddy")

    form = ttk.Frame(window, padding=20)
    form.pack(fill="both", expand=True)

    ttk.Label(
        form,
        text="Change the initial interest rate to see a prediction for a sample loan.",
        wraplength=360,
    ).pack(anchor="w")

    ttk.Label(form, text="Initial interest rate (%)").pack(
        anchor="w", pady=(12, 0)
    )
    rate_entry = ttk.Entry(form)
    rate_entry.insert(0, str(sample_loan["InitialInterestRate"]))
    rate_entry.pack(fill="x")

    result = tk.StringVar(master=window)

    def show_prediction():
        try:
            rate = float(rate_entry.get())
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError
        except ValueError:
            result.set("Enter a positive interest rate, such as 8.5.")
            return

        loan = {**sample_loan, "InitialInterestRate": rate}
        predicted_class, score = predict_loan(loan, model, preprocessor)
        label = "Default" if predicted_class == 1 else "No default"
        result.set(f"Prediction: {label}\nModel default score: {score:.1%}")

    ttk.Button(form, text="Predict", command=show_prediction).pack(
        anchor="w", pady=10
    )
    ttk.Label(form, textvariable=result).pack(anchor="w")

    rate_entry.focus_set()
    window.mainloop()


if __name__ == "__main__":
    main()