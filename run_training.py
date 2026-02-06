# run_training.py
"""
This script trains all classification models once and
prints a consolidated comparison table.
The results will be reused in README and Streamlit UI.
"""

import pandas as pd

from model.data_loader import load_and_split_data
from model.logistic_model import train_logistic_regression
from model.decision_tree_model import train_decision_tree
from model.knn_model import train_knn
from model.naive_bayes_model import train_naive_bayes
from model.random_forest_model import train_random_forest
from model.xgboost_model import train_xgboost


def run_all_models(dataset_path):
    X_train, X_test, y_train, y_test = load_and_split_data(dataset_path)

    results = []

    model_runs = [
        ("Logistic Regression", train_logistic_regression),
        ("Decision Tree", train_decision_tree),
        ("KNN", train_knn),
        ("Naive Bayes", train_naive_bayes),
        ("Random Forest", train_random_forest),
        ("XGBoost", train_xgboost),
    ]

    for model_name, train_fn in model_runs:
        print(f"Training {model_name}...")
        _, metrics = train_fn(X_train, X_test, y_train, y_test)

        metrics["Model"] = model_name
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]

    return results_df


if __name__ == "__main__":
    dataset_csv = "credit_card_default.csv"  # <-- your dataset file name
    metrics_table = run_all_models(dataset_csv)

    print("\nFinal Model Comparison Table:\n")
    print(metrics_table.to_string(index=False))
