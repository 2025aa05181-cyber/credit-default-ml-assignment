import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix
)

# --------------------------------------------------
# Config
# --------------------------------------------------
TARGET_COL = "default.payment.next.month"

MODEL_FILES = {
    "Logistic Regression": "saved_models/logistic.joblib",
    "Decision Tree": "saved_models/decision_tree.joblib",
    "kNN": "saved_models/knn.joblib",
    "Naive Bayes": "saved_models/naive_bayes.joblib",
    "Random Forest": "saved_models/random_forest.joblib",
    "XGBoost": "saved_models/xgboost.joblib"
}

st.title("Credit Card Default Prediction – Evaluation App")

# --------------------------------------------------
# Upload CSV
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload TEST dataset (must include target column)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

if TARGET_COL not in df.columns:
    st.error(
        f"Uploaded CSV must contain target column: {TARGET_COL}"
    )
    st.stop()

X_test = df.drop(columns=[TARGET_COL])
y_true = df[TARGET_COL]

# --------------------------------------------------
# Model selection
# --------------------------------------------------
model_name = st.selectbox(
    "Select Model",
    list(MODEL_FILES.keys())
)

model = joblib.load(MODEL_FILES[model_name])

# --------------------------------------------------
# Prediction
# --------------------------------------------------
y_pred = model.predict(X_test)

if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_true, y_prob)
else:
    auc = "NA"

# --------------------------------------------------
# Metrics
# --------------------------------------------------
metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "Precision": precision_score(y_true, y_pred),
    "Recall": recall_score(y_true, y_pred),
    "F1 Score": f1_score(y_true, y_pred),
    "MCC": matthews_corrcoef(y_true, y_pred),
    "AUC": auc
}

st.subheader("Evaluation Metrics (Computed on Uploaded Data)")
st.table(pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"]))

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)
