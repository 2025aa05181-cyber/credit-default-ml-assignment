# run_training.py
"""
Train all ML models once on the Credit Card Default dataset
and save the trained models for Streamlit inference.
"""

import os
import joblib
import pandas as pd

from model.data_loader import load_and_split_data
from model.logistic_model import train_logistic_regression
from model.decision_tree_model import train_decision_tree
from model.knn_model import train_knn
from model.naive_bayes_model import train_naive_bayes
from model.random_forest_model import train_random_forest
from model.xgboost_model import train_xgboost


# --------------------------------------------------
# Create directory to store trained models
# --------------------------------------------------
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    # ----------------------------------------------
    # 1. Load dataset and split
    # ----------------------------------------------
    dataset_path = "credit_card_default.csv"
    X_train, X_test, y_train, y_test = load_and_split_data(dataset_path)

    # ----------------------------------------------
    # 2. Train models one by one
    # ----------------------------------------------
    models = {
        "logistic": train_logistic_regression,
        "decision_tree": train_decision_tree,
        "knn": train_knn,
        "naive_bayes": train_naive_bayes,
        "random_forest": train_random_forest,
        "xgboost": train_xgboost,
    }

    results = []

    for model_name, train_func in models.items():
        print(f"\nTraining {model_name} model...")
        trained_model, metrics = train_func(
            X_train, X_test, y_train, y_test
        )

        # ------------------------------------------
        # 3. SAVE trained model
        # ------------------------------------------
        model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
        joblib.dump(trained_model, model_path)

        print(f"Saved model to: {model_path}")
        results.append({"Model": model_name, **metrics})

    # ----------------------------------------------
    # 4. Print metrics table (for your reference)
    # ----------------------------------------------
    metrics_df = pd.DataFrame(results)
    print("\nFinal Model Metrics:\n")
    print(metrics_df)


if __name__ == "__main__":
    main()
