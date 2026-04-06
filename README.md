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
* Feature engineering (tenure groups, spending behavior)

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

```
customer-churn-prediction
├── app.py
├── train.py
├── data/
│   └── Churn.csv
├── static/
│   └── churn.html
├── artifacts/
│   ├── best_model.joblib
│   └── metadata.json
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Train the model

```
python train.py --data data/Churn.csv
```

### 3. Run the API server

```
uvicorn app:app --reload
```

### 4. Open dashboard

Open the file in your browser:

```
static/churn.html
```

Dashboard file:

* `churn.html` 

---

## API Endpoint

POST request:

```
http://127.0.0.1:8000/predict
```

Backend implementation:

* `app.py` 

---

## Output

* Prediction: Churn or Stay
* Probability scores
* Feature impact (SHAP values)
* Human-readable explanation

---

## Author

Dhruv Vats,Navoshma Mittal
BTech CSE, UPES
