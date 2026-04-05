# Traffic Volume Estimation

A machine learning project to predict highway traffic volume based on weather, date, and time features.

## Project Structure

```
traffic_volume_project/
├── notebooks/
│   ├── Traffic_Volume.ipynb               # Data prep, training, model export
│   └── trafficVolume_ibm_scoring_end_point.ipynb  # IBM Watson ML scoring
├── flask/
│   ├── app.py                             # Flask web application
│   ├── model.pkl                          # Generated after running notebook
│   ├── scaler.pkl                         # Generated after running notebook
│   ├── encoder_weather.pkl                # Generated after running notebook
│   ├── encoder_holiday.pkl                # Generated after running notebook
│   └── templates/
│       ├── index.html                     # Input form
│       └── result.html                    # Prediction result page
├── data/
│   └── TrafficVolume.csv                  # Raw dataset
└── requirement.txt                        # Python dependencies
```

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirement.txt
```

### 2. Train the model
Open and run `notebooks/Traffic_Volume.ipynb` from top to bottom.
This will generate the `.pkl` files inside the `flask/` folder.

### 3. Run the Flask app
```bash
cd flask
python app.py
```
Then visit `http://127.0.0.1:5000` in your browser.

## Models Compared
| Model              | Test R² |
|--------------------|---------|
| Linear Regression  | ~0.10   |
| Decision Tree      | ~0.69   |
| **Random Forest**  | **~0.80** ✅ |
| SVR                | ~0.10   |
| XGBoost            | ~0.80   |

Random Forest is saved as the production model.
