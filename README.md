#  Customer Churn Prediction

##  Problem

Businesses lose customers (churn), which affects revenue.
This project predicts whether a customer will leave or stay using machine learning techniques.

---

##  Model Used

* Random Forest Classifier
* SMOTE & SMOTE-Tomek (for handling class imbalance)

---

##  Dataset

* Telco Customer Churn Dataset
* Features include: tenure, services, monthly charges, total charges, etc.

---

##  Model Optimization

* Compared multiple imbalance strategies:

  * No balancing
  * SMOTE
  * SMOTE-Tomek
* Automatically selected the best model based on:

  * F1 Score
  * Recall
  * ROC-AUC

---

##  Results (Best Model)

* Accuracy: 0.79
* Precision: 0.64
* Recall: 0.59
* F1 Score: 0.60
* ROC-AUC: 0.84

---

##  Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* Streamlit

---

##  How to Run

```bash
pip install -r requirements.txt
python train.py --data Churn.csv
streamlit run app.py
```

---

##  Output

* Best trained model saved using Joblib
* Comparison results saved as CSV
* Evaluation report generated automatically
* Feature names and metadata stored

---

##  Key Features

* End-to-end ML pipeline
* Handles missing data and feature engineering
* Supports categorical encoding and preprocessing
* Automatically selects best strategy
* Deployable using Streamlit

---

##  Future Improvements

* Add SHAP for explainable AI
* Improve accuracy using XGBoost / Deep Learning
* Deploy on cloud (AWS / Render / Hugging Face)

---

##  Author

Navoshma Mittal
Dhruv Vats
BTech CSE | Data Science Enthusiast
