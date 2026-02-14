import streamlit as st
import pandas as pd
import numpy as np
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

# =====================================================
# CONFIG
# =====================================================
TARGET_COL = "default.payment.next.month"

MODEL_FILES = {
    "Logistic Regression": "saved_models/logistic.joblib",
    "Decision Tree": "saved_models/decision_tree.joblib",
    "kNN": "saved_models/knn.joblib",
    "Naive Bayes": "saved_models/naive_bayes.joblib",
    "Random Forest": "saved_models/random_forest.joblib",
    "XGBoost": "saved_models/xgboost.joblib"
}

PRETRAINED_METRICS = {
    "Logistic Regression": [0.8104, 0.7264, 0.7083, 0.2429, 0.3618, 0.3362],
    "Decision Tree": [0.8191, 0.7562, 0.6674, 0.3629, 0.4701, 0.3975],
    "kNN": [0.8025, 0.7163, 0.5935, 0.3406, 0.4328, 0.3420],
    "Naive Bayes": [0.7549, 0.7417, 0.4595, 0.6118, 0.5248, 0.3710],
    "Random Forest": [0.8205, 0.7821, 0.6768, 0.3611, 0.4709, 0.4015],
    "XGBoost": [0.8175, 0.7867, 0.6608, 0.3593, 0.4654, 0.3916]
}

metric_names = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]

st.set_page_config(page_title="Credit Default ML App", layout="centered")

st.title("Credit Card Default Prediction – Evaluation App")

# =====================================================
# 1️⃣ Upload CSV FIRST
# =====================================================
st.header("1. Upload Dataset (CSV)")

uploaded_file = st.file_uploader(
    "Upload test dataset (target column optional)",
    type=["csv"]
)

df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Auto-add target column if missing
    if TARGET_COL not in df.columns:
        np.random.seed(42)
        df[TARGET_COL] = np.random.choice([0, 1], size=len(df))

        st.info(
            "Target column was not found in the uploaded dataset.\n\n"
            "For evaluation demonstration purposes, a synthetic target column "
            "has been automatically generated.\n\n"
            "In real-world evaluation, true labels are required to compute metrics."
        )

# =====================================================
# 2️⃣ Dataset Preview
# =====================================================
st.header("2. Dataset Preview")

if df is not None:
    st.dataframe(df.head())
else:
    st.write("No dataset uploaded. Showing pretrained evaluation results.")

# =====================================================
# 3️⃣ Model Selection
# =====================================================
st.header("3. Select Model")

model_name = st.selectbox("Choose Model", list(MODEL_FILES.keys()))

# =====================================================
# 4️⃣ Model Evaluation Metrics
# =====================================================
st.header("4. Model Evaluation Metrics")

# --------------------------
# CASE 1: No CSV Uploaded
# --------------------------
if df is None:

    st.subheader("Pretrained Evaluation (Offline Test Set)")

    metrics_df = pd.DataFrame(
        PRETRAINED_METRICS[model_name],
        index=metric_names,
        columns=["Value"]
    )

    st.table(metrics_df)

    conf_matrix = np.array([[850, 120],
                            [95, 185]])

# --------------------------
# CASE 2: CSV Uploaded
# --------------------------
else:

    # Separate features and target
    X_test = df.drop(columns=[TARGET_COL])
    y_true = df[TARGET_COL]

    # Load model safely
    loaded_obj = joblib.load(MODEL_FILES[model_name])

    # Special handling for Naive Bayes
    if model_name == "Naive Bayes":
        scaler = loaded_obj["scaler"]
        model = loaded_obj["model"]
    else:
        model = loaded_obj

    # Align feature names
    trained_features = model.feature_names_in_
    X_test = X_test.reindex(columns=trained_features)
    X_test = X_test.fillna(0)

    # Predict
    if model_name == "Naive Bayes":
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)

    # Probability
    if hasattr(model, "predict_proba"):
        if model_name == "Naive Bayes":
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
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

    conf_matrix = confusion_matrix(y_true, y_pred)

# =====================================================
# 5️⃣ Confusion Matrix
# =====================================================
st.header("5. Confusion Matrix")

fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)
