# Customer Churn Prediction

## Project Overview

This project predicts whether a customer will churn (leave) or stay using Machine Learning.
It also provides explainable insights using SHAP and an interactive dashboard for visualization.

---

## Features

* Customer churn prediction using a trained ML model
* SHAP-based explainability (feature impact)
* Interactive dashboard using HTML and Chart.js
* FastAPI backend for real-time predictions
* Feature engineering (tenure, spending behavior)

---

## Model Details

* Algorithm: Random Forest
* Imbalance Handling: SMOTE / SMOTETomek
* Pipeline: Preprocessing + Model

Training implementation:

* `train.py`

---

## Dataset

* Telco Customer Churn Dataset
* Includes:

  * Customer tenure
  * Services used
  * Monthly and total charges
  * Contract type

---

## Project Structure

```text
customer-churn-prediction
├── app.py
├── train.py
├── data/
│   └── Churn.csv
├── static/
│   └── churn.html
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Generate model files (IMPORTANT)

Before running the API, you must generate model files:

```bash
python train.py --data data/Churn.csv
```

This will create:

```text
artifacts/
├── best_model.joblib
├── metadata.json
```

---

### 3. Run the API server

```bash
uvicorn app:app --reload
```

---

### 4. Open dashboard

Open this file in your browser:

```text
static/churn.html
```

---

## API Endpoint

POST request:

```text
http://127.0.0.1:8000/predict
```

---

## Output

* Prediction: Churn or Stay
* Probability scores
* Feature impact (SHAP values)
* Human-readable explanation

---

## Notes

* The `artifacts/` folder is generated automatically and is not included in the repository.
* Make sure to run the training step before starting the API.

---

## Author

Navoshma Mittal,Dhruv Vats
BTech CSE, UPES
