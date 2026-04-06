from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import json
import shap
import numpy as np

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("artifacts/best_model.joblib")

# Load metadata
with open("artifacts/metadata.json") as f:
    metadata = json.load(f)

# Extract pipeline parts
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# SHAP explainer
explainer = shap.Explainer(classifier)

# Default values
DEFAULTS = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 0,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 0,
    "TotalCharges": 0,
}


def compute_features(data):
    tenure = float(data.get("tenure", 0))
    monthly = float(data.get("MonthlyCharges", 0))
    total = float(data.get("TotalCharges", 0))

    data["AvgCharges"] = total / (tenure + 1)
    data["HighSpender"] = 1 if monthly > 70 else 0
    data["IsNewCustomer"] = 1 if tenure < 6 else 0

    if tenure <= 12:
        data["TenureGroup"] = "0-1yr"
    elif tenure <= 24:
        data["TenureGroup"] = "1-2yr"
    elif tenure <= 48:
        data["TenureGroup"] = "2-4yr"
    else:
        data["TenureGroup"] = "4+yr"

    return data


def normalize_input(user_data):
    merged = DEFAULTS.copy()
    merged.update(user_data or {})
    merged = compute_features(merged)

    ordered = {}
    for col in metadata["all_input_columns"]:
        ordered[col] = merged.get(col, DEFAULTS.get(col, None))

    return ordered


# ---------- HUMAN EXPLANATION ----------
def generate_explanations(shap_list):
    explanations = []

    for item in shap_list:
        feature = item["feature"]
        impact = item["impact"]

        if impact > 0:
            if "Fiber optic" in feature:
                explanations.append("Customer uses fiber internet which increases churn risk")
            elif "Month-to-month" in feature:
                explanations.append("Customer has a short-term contract, so they may leave anytime")
            elif "AvgCharges" in feature:
                explanations.append("Customer is paying high charges")
            elif "OnlineSecurity_No" in feature:
                explanations.append("Customer does not have online security")
            elif "TechSupport_No" in feature:
                explanations.append("Customer does not have technical support")
            elif "Electronic check" in feature:
                explanations.append("Customer uses electronic payment method")
            elif "IsNewCustomer" in feature:
                explanations.append("Customer is new and less loyal")
            elif "tenure" in feature:
                explanations.append("Customer has low experience with company")

        else:
            if "DSL" in feature:
                explanations.append("Customer uses stable internet service")
            elif "tenure" in feature:
                explanations.append("Customer is loyal and long-term user")

    return explanations[:5]


# ---------- API ----------
@app.post("/predict")
def predict(data: dict):
    try:
        clean_data = normalize_input(data)
        df = pd.DataFrame([clean_data])

        prob = float(model.predict_proba(df)[0][1])

        shap_list = []

        try:
            transformed = preprocessor.transform(df)

            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()

            shap_values = explainer(transformed)

            feature_names = preprocessor.get_feature_names_out()
            values = shap_values.values[0]

            for name, val in zip(feature_names, values):
                impact_value = float(np.array(val).flatten()[0])

                shap_list.append({
                    "feature": str(name),
                    "impact": impact_value
                })

            shap_list = sorted(
                shap_list,
                key=lambda x: abs(x["impact"]),
                reverse=True
            )[:10]

        except Exception as e:
            print("SHAP ERROR:", e)
            shap_list = []

        explanations = generate_explanations(shap_list)

        return {
            "success": True,
            "prediction": "Churn" if prob > 0.35 else "Stay",
            "churn_probability": prob,
            "stay_probability": 1 - prob,
            "shap": shap_list,
            "explanations": explanations
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
