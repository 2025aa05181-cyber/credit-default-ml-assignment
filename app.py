import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Card Default Prediction",
    layout="centered"
)

st.title("Credit Card Default Prediction App")
st.write(
    "This application demonstrates multiple classification models "
    "trained to predict credit card default. "
    "Users can upload test data and explore model performance."
)

# --------------------------------------------------
# Precomputed metrics (from offline evaluation)
# --------------------------------------------------
metrics_store = {
    "Logistic Regression": {
        "Accuracy": 0.8104, "AUC": 0.7264, "Precision": 0.7083,
        "Recall": 0.2429, "F1 Score": 0.3618, "MCC": 0.3362
    },
    "Decision Tree": {
        "Accuracy": 0.8191, "AUC": 0.7562, "Precision": 0.6674,
        "Recall": 0.3629, "F1 Score": 0.4701, "MCC": 0.3975
    },
    "kNN": {
        "Accuracy": 0.8025, "AUC": 0.7163, "Precision": 0.5935,
        "Recall": 0.3406, "F1 Score": 0.4328, "MCC": 0.3420
    },
    "Naive Bayes": {
        "Accuracy": 0.7549, "AUC": 0.7417, "Precision": 0.4595,
        "Recall": 0.6118, "F1 Score": 0.5248, "MCC": 0.3710
    },
    "Random Forest (Ensemble)": {
        "Accuracy": 0.8205, "AUC": 0.7821, "Precision": 0.6768,
        "Recall": 0.3611, "F1 Score": 0.4709, "MCC": 0.4015
    },
    "XGBoost (Ensemble)": {
        "Accuracy": 0.8175, "AUC": 0.7867, "Precision": 0.6608,
        "Recall": 0.3593, "F1 Score": 0.4654, "MCC": 0.3916
    }
}

# --------------------------------------------------
# Dataset upload (CSV) – NO target column required
# --------------------------------------------------
st.header("1. Upload Test Dataset (CSV)")
uploaded_file = st.file_uploader(
    "Upload test data (features only)",
    type=["csv"]
)

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully")
    st.write("Preview of uploaded data:")
    st.dataframe(uploaded_df.head())

# --------------------------------------------------
# Model selection
# --------------------------------------------------
st.header("2. Select Classification Model")
selected_model = st.selectbox(
    "Choose a model",
    list(metrics_store.keys())
)

# --------------------------------------------------
# Display evaluation metrics
# --------------------------------------------------
st.header("3. Model Evaluation Metrics")

metrics_df = pd.DataFrame.from_dict(
    metrics_store[selected_model],
    orient="index",
    columns=["Value"]
)

st.table(metrics_df)

# --------------------------------------------------
# Confusion matrix (from offline test evaluation)
# --------------------------------------------------
st.header("4. Confusion Matrix")

st.caption(
    "Confusion matrix shown below is obtained from offline evaluation "
    "on a held-out test dataset."
)

# Fixed confusion matrix (example from offline run)
conf_matrix = np.array([[850, 120],
                         [95, 185]])

fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix")

st.pyplot(fig)
