import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Credit Card Default Prediction App")

# -------------------------------
# Upload test dataset
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload test dataset (CSV with target column)",
    type=["csv"]
)

# -------------------------------
# Model selection
# -------------------------------
model_name = st.selectbox(
    "Select Classification Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    target_col = "default.payment.next.month"

    if target_col not in df.columns:
        st.error("Uploaded CSV must contain target column: default.payment.next.month")
        st.stop()

    X_test = df.drop(columns=[target_col])
    y_true = df[target_col]

    # -------------------------------
    # Load trained model
    # -------------------------------
    model = joblib.load(f"saved_models/{model_name}.joblib")

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = "NA"

    # -------------------------------
    # Compute metrics dynamically
    # -------------------------------
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUC": auc
    }

    st.subheader("Model Evaluation Metrics (Computed on Uploaded CSV)")
    st.table(pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"]))

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
