from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)

def compute_classification_metrics(true_labels, predicted_labels, predicted_probs):
    return {
        "Accuracy": accuracy_score(true_labels, predicted_labels),
        "Precision": precision_score(true_labels, predicted_labels),
        "Recall": recall_score(true_labels, predicted_labels),
        "F1": f1_score(true_labels, predicted_labels),
        "AUC": roc_auc_score(true_labels, predicted_probs),
        "MCC": matthews_corrcoef(true_labels, predicted_labels)
    }