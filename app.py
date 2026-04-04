from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ARTIFACT_DIR = Path("artifacts")
THRESHOLD = 0.35

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)


@st.cache_resource
def load_assets():
    model = joblib.load(ARTIFACT_DIR / "best_model.joblib")

    with open(ARTIFACT_DIR / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    sample_inputs = pd.read_csv(ARTIFACT_DIR / "sample_inputs.csv")
    return model, metadata, sample_inputs


def compute_extra_features(data: dict) -> dict:
    tenure = float(data.get("tenure", 0) or 0)
    monthly = float(data.get("MonthlyCharges", 0) or 0)
    total = float(data.get("TotalCharges", 0) or 0)

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


def build_input_form(sample_inputs: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    st.sidebar.header("Customer Input")

    if sample_inputs.empty:
        st.error("sample_inputs.csv is empty.")
        st.stop()

    first_row = sample_inputs.iloc[0].to_dict()
    data = {}

    allowed_columns = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges"
    ]

    for col in metadata["all_input_columns"]:
        if col not in allowed_columns:
            continue

        if col in metadata["numeric_columns"]:
            default_value = first_row.get(col, 0)
            if pd.isna(default_value):
                default_value = 0

            data[col] = st.sidebar.number_input(
                label=col,
                value=float(default_value)
            )
        else:
            options = (
                sample_inputs[col].dropna().astype(str).unique().tolist()
                if col in sample_inputs.columns else []
            )
            options = sorted(options)

            if not options:
                options = [str(first_row.get(col, ""))]

            default_value = str(first_row.get(col, options[0]))
            default_index = options.index(default_value) if default_value in options else 0

            data[col] = st.sidebar.selectbox(
                label=col,
                options=options,
                index=default_index
            )

    data = compute_extra_features(data)

    ordered_data = {}
    for col in metadata["all_input_columns"]:
        ordered_data[col] = data.get(col, first_row.get(col, None))

    return pd.DataFrame([ordered_data])


def generate_reasons(input_df: pd.DataFrame, probability: float) -> tuple[list[str], list[str]]:
    row = input_df.iloc[0]

    churn_reasons = []
    stay_reasons = []

    contract = str(row.get("Contract", "")).strip().lower()
    payment = str(row.get("PaymentMethod", "")).strip().lower()
    paperless = str(row.get("PaperlessBilling", "")).strip().lower()
    tenure = float(row.get("tenure", 0) or 0)
    monthly = float(row.get("MonthlyCharges", 0) or 0)
    internet = str(row.get("InternetService", "")).strip().lower()
    tech = str(row.get("TechSupport", "")).strip().lower()
    security = str(row.get("OnlineSecurity", "")).strip().lower()

    if contract == "month-to-month":
        churn_reasons.append("Customer has a month-to-month contract, which usually means lower commitment.")
    elif contract in ["one year", "two year"]:
        stay_reasons.append("Customer has a long-term contract, which usually means higher commitment.")

    if payment == "electronic check":
        churn_reasons.append("Electronic check payment is often linked with higher churn risk.")
    else:
        stay_reasons.append("Payment method looks more stable than electronic check.")

    if paperless == "yes":
        churn_reasons.append("Paperless billing can be associated with slightly higher churn behavior.")
    else:
        stay_reasons.append("Customer is not using paperless billing, which may indicate more stable behavior.")

    if tenure < 6:
        churn_reasons.append("Customer is new, and new customers are more likely to leave early.")
    elif tenure > 24:
        stay_reasons.append("Customer has stayed for a long time, which suggests loyalty.")

    if monthly > 80:
        churn_reasons.append("Monthly charges are high, which may increase the chance of churn.")
    elif monthly < 50:
        stay_reasons.append("Monthly charges are moderate, which may help retain the customer.")

    if internet == "fiber optic":
        churn_reasons.append("Fiber optic customers sometimes show higher churn in churn datasets.")

    if tech == "no":
        churn_reasons.append("Customer does not have tech support, which may increase dissatisfaction.")
    else:
        stay_reasons.append("Customer has tech support, which may improve retention.")

    if security == "no":
        churn_reasons.append("Customer does not have online security, which may reduce service stickiness.")
    else:
        stay_reasons.append("Customer has online security, which can improve retention.")

    if probability >= THRESHOLD:
        if not churn_reasons:
            churn_reasons.append("Overall input pattern is closer to churn-risk customers.")
    else:
        if not stay_reasons:
            stay_reasons.append("Overall input pattern is closer to customers who usually stay.")

    return churn_reasons[:4], stay_reasons[:4]


def show_feature_importance(model):
    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        feature_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(10)

        st.subheader("Top 10 Important Features")
        st.dataframe(importance_df, use_container_width=True)
        st.bar_chart(importance_df.set_index("Feature"))

    except Exception as e:
        st.warning(f"Could not display feature importance: {e}")


def main():
    st.title("📉 Customer Churn Prediction App")
    st.write("This app predicts whether a customer is likely to churn using the trained Random Forest model.")

    if not ARTIFACT_DIR.exists():
        st.error("Artifacts folder not found. Please run train.py first.")
        st.stop()

    required_files = [
        ARTIFACT_DIR / "best_model.joblib",
        ARTIFACT_DIR / "metadata.json",
        ARTIFACT_DIR / "sample_inputs.csv"
    ]

    missing_files = [str(file) for file in required_files if not file.exists()]
    if missing_files:
        st.error("Missing required files:\n" + "\n".join(missing_files))
        st.stop()

    model, metadata, sample_inputs = load_assets()

    st.subheader("Enter Customer Details")
    input_df = build_input_form(sample_inputs, metadata)

    display_columns = [col for col in [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges"
    ] if col in input_df.columns]

    st.dataframe(input_df[display_columns], use_container_width=True)

    if st.button("Predict Churn", type="primary"):
        try:
            probability = float(model.predict_proba(input_df)[0][1])
            prediction = 1 if probability >= THRESHOLD else 0

            churn_reasons, stay_reasons = generate_reasons(input_df, probability)

            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("Customer is likely to churn.")
            else:
                st.success("Customer is likely to stay.")

            st.write(f"**Churn Probability:** {probability:.2%}")
            st.write(f"**Stay Probability:** {(1 - probability):.2%}")
            st.write(f"**Decision Threshold Used:** {THRESHOLD:.2f}")

            st.subheader("Reason for Prediction")

            if prediction == 1:
                st.markdown("### Why customer may churn")
                for reason in churn_reasons:
                    st.write(f"• {reason}")

                if stay_reasons:
                    st.markdown("### Positive signs")
                    for reason in stay_reasons[:2]:
                        st.write(f"• {reason}")
            else:
                st.markdown("### Why customer may stay")
                for reason in stay_reasons:
                    st.write(f"• {reason}")

                if churn_reasons:
                    st.markdown("### Risk signs")
                    for reason in churn_reasons[:2]:
                        st.write(f"• {reason}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    with st.expander("Show Model Feature Importance"):
        show_feature_importance(model)


if __name__ == "__main__":
    main()
