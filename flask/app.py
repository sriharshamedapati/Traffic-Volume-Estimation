import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load saved model artifacts
model      = pickle.load(open("model.pkl",            "rb"))
scaler     = pickle.load(open("scaler.pkl",           "rb"))
le_weather = pickle.load(open("encoder_weather.pkl",  "rb"))
le_holiday = pickle.load(open("encoder_holiday.pkl",  "rb"))


def encode_holiday(holiday_str):
    """
    Encode holiday safely.
    The LabelEncoder was trained on a column that had real NaN values (float),
    so le_holiday.classes_ contains np.nan — NOT the string 'nan'.
    We must find NaN's index manually instead of calling .transform() on it.
    """
    if not holiday_str:
        # Find the index of the NaN entry in classes_
        for i, cls in enumerate(le_holiday.classes_):
            try:
                if pd.isna(cls):
                    return i
            except TypeError:
                pass
        return 0  # fallback if no NaN class found

    # Normal named holiday — use transform as usual
    return le_holiday.transform([holiday_str])[0]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ── Read form values ───────────────────────────────────────────────────
        holiday = request.form.get("holiday", "").strip()
        weather = request.form.get("weather", "").strip()
        temp    = request.form.get("temp",    "").strip()
        rain    = request.form.get("rain",    "").strip()
        snow    = request.form.get("snow",    "").strip()
        year    = request.form.get("year",    "").strip()
        month   = request.form.get("month",   "").strip()
        day     = request.form.get("day",     "").strip()
        hours   = request.form.get("hours",   "").strip()
        minutes = request.form.get("minutes", "").strip()
        seconds = request.form.get("seconds", "").strip()

        # ── Validate weather (required) ────────────────────────────────────────
        if not weather:
            return render_template(
                "index.html",
                prediction_text="Please select a Weather condition before predicting."
            )

        # ── Validate named holiday against known classes (skip NaN check) ──────
        if holiday:
            known_holidays = [c for c in le_holiday.classes_ if not _is_nan(c)]
            if holiday not in known_holidays:
                return render_template(
                    "index.html",
                    prediction_text=f"Unknown holiday value '{holiday}'."
                )

        # ── Validate weather against known classes ─────────────────────────────
        if weather not in list(le_weather.classes_):
            return render_template(
                "index.html",
                prediction_text=f"Unknown weather value '{weather}'."
            )

        # ── Convert numeric fields ─────────────────────────────────────────────
        temp    = float(temp)
        rain    = float(rain)
        snow    = float(snow)
        year    = int(year)
        month   = int(month)
        day     = int(day)
        hours   = int(hours)
        minutes = int(minutes)
        seconds = int(seconds)

        # ── Encode categoricals ────────────────────────────────────────────────
        holiday_encoded = encode_holiday(holiday)
        weather_encoded = le_weather.transform([weather])[0]

        # ── Build input DataFrame ──────────────────────────────────────────────
        input_df = pd.DataFrame(
            [[holiday_encoded, temp, rain, snow, weather_encoded,
              day, month, year, hours, minutes, seconds]],
            columns=["holiday", "temp", "rain", "snow", "weather",
                     "day", "month", "year", "hours", "minutes", "seconds"]
        )

        # ── Scale & predict ────────────────────────────────────────────────────
        input_scaled = scaler.transform(input_df)
        prediction   = model.predict(input_scaled)
        volume       = int(round(prediction[0]))

        return render_template(
            "result.html",
            prediction_text=f"Estimated Traffic Volume: {volume} vehicles/hour"
        )

    except ValueError as e:
        return render_template(
            "index.html",
            prediction_text=f"Invalid input - please check all fields are filled correctly. ({e})"
        )
    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


def _is_nan(val):
    try:
        return pd.isna(val)
    except TypeError:
        return False


if __name__ == "__main__":
    app.run(debug=True)