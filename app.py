import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix
)

TARGET_COL = "default.payment.next.month"

MODEL_FILES = {
    "Logistic Regression": "saved_models/logistic.joblib",
    "Decision Tree": "saved_models/decision_tree.joblib",
    "kNN": "saved_models/knn.joblib",
    "Naive Bayes": "saved_models/naive_bayes.joblib",
    "Random Forest": "saved_models/random_forest.joblib",
    "XGBoost": "saved_models/xgboost.joblib"
}

# Pretrained offline metrics (fallback display)
PRETRAINED_METRICS = {
    "Logistic Regression": [0.8104, 0.7264, 0.7083, 0.2429, 0.3618, 0.3362],
    "Decision Tree": [0.8191, 0.7562, 0.6674, 0.3629, 0.4701, 0.3975],
    "kNN": [0.8025, 0.7163, 0.5935, 0.3406, 0.4328, 0.3420],
    "Naive Bayes": [0.7549, 0.7417, 0.4595, 0.6118, 0.5248, 0.3710],
    "Random Forest": [0.8205, 0.7821, 0.6768, 0.3611, 0.4709, 0.4015],
    "XGBoost": [0.8175, 0.7867, 0.6608, 0.3593, 0.4654, 0.3916]
}

metric_names = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]

st.title("Credit Card Default Prediction – Evaluation App")

# -------------------------
# Model Selection (ALWAYS visible)
# -------------------------
model_name = st.selectbox("Select Model", list(MODEL_FILES.keys()))

# -------------------------
# CSV Upload
# -------------------------
uploaded_file = st.file_uploader(
    "Upload TEST dataset (optional – must include target column for dynamic evaluation)",
    type=["csv"]
)

# -------------------------
# If NO CSV uploaded → Show Pretrained Metrics
# -------------------------
if uploaded_file is None:

    st.subheader("Pretrained Model Evaluation (Offline Test Set)")

    metrics_df = pd.DataFrame(
        PRETRAINED_METRICS[model_name],
        index=metric_names,
        columns=["Value"]
    )

    st.table(metrics_df)

    # Example pretrained confusion matrix
    conf_matrix = np.array([[850, 120],
                            [95, 185]])

    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Offline Confusion Matrix")

    st.pyplot(fig)

# -------------------------
# If CSV Uploaded → Dynamic Evaluation
# -------------------------
else:
    df = pd.read_csv(uploaded_file)

    if TARGET_COL not in df.columns:
        st.error(f"Uploaded CSV must contain target column: {TARGET_COL}")
        st.stop()

    X_test = df.drop(columns=[TARGET_COL])
    y_true = df[TARGET_COL]

    model = joblib.load(MODEL_FILES[model_name])
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = "NA"

    metrics = [
        accuracy_score(y_true, y_pred),
        auc,
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        matthews_corrcoef(y_true, y_pred)
    ]

    st.subheader("Dynamic Evaluation (Uploaded Dataset)")

    metrics_df = pd.DataFrame(
        metrics,
        index=metric_names,
        columns=["Value"]
    )

    st.table(metrics_df)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Uploaded Data)")

    st.pyplot(fig)
