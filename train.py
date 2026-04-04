from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_CANDIDATES = ["Churn", "churn", "Exited", "Target", "target"]


def find_target_column(df: pd.DataFrame) -> str:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"Target column not found. Expected one of: {TARGET_CANDIDATES}")


def clean_telco_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df = df.replace(r"^\s*$", np.nan, regex=True)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["AvgCharges"] = df["TotalCharges"] / (df["tenure"] + 1)

    if "MonthlyCharges" in df.columns:
        df["HighSpender"] = (df["MonthlyCharges"] > 70).astype(int)

    if "tenure" in df.columns:
        df["IsNewCustomer"] = (df["tenure"] < 6).astype(int)
        df["TenureGroup"] = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, np.inf],
            labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"]
        ).astype(str)

    return df


def encode_target(y: pd.Series) -> pd.Series:
    if y.dtype == object:
        y = y.astype(str).str.strip().str.lower()
        mapping = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
        mapped = y.map(mapping)
        if mapped.isna().any():
            raise ValueError("Target column contains values that cannot be encoded to 0/1.")
        return mapped.astype(int)
    return y.astype(int)


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols


def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def train_all_strategies(df: pd.DataFrame, output_dir: Path) -> None:
    target_col = find_target_column(df)

    X = df.drop(columns=[target_col])
    X = add_features(X)

    y = encode_target(df[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    strategies = {
        "none": None,
        "smote": SMOTE(random_state=42),
        "smotetomek": SMOTETomek(random_state=42),
    }

    results = []
    models = {}

    for strategy_name, sampler in strategies.items():
        if sampler is None:
            model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", rf),
                ]
            )
        else:
            model = ImbPipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("sampler", sampler),
                    ("classifier", rf),
                ]
            )

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        metrics["strategy"] = strategy_name
        results.append(metrics)
        models[strategy_name] = model

    results_df = pd.DataFrame(results).sort_values(
        by=["f1_score", "recall", "roc_auc", "accuracy"], ascending=False
    )

    best_strategy = results_df.iloc[0]["strategy"]
    best_model = models[best_strategy]

    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "comparison_results.csv", index=False)
    joblib.dump(best_model, output_dir / "best_model.joblib")

    metadata = {
        "target_column": target_col,
        "best_strategy": best_strategy,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "all_input_columns": X.columns.tolist(),
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    X_train.head(50).to_csv(output_dir / "sample_inputs.csv", index=False)

    preprocessor_fitted = best_model.named_steps["preprocessor"]
    feature_names = preprocessor_fitted.get_feature_names_out().tolist()
    with open(output_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    print("Training complete.")
    print(f"Best strategy: {best_strategy}")
    print()
    print(results_df.to_string(index=False))
    print()

    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()

    with open(output_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("Best strategy: " + best_strategy + "\n\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--output", default="artifacts", help="Directory to save model files")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df = clean_telco_dataframe(df)

    train_all_strategies(df, Path(args.output))


if __name__ == "__main__":
    main()
