from sklearn.tree import DecisionTreeClassifier
from model.metrics_helper import compute_classification_metrics

def train_decision_tree(X_train, X_test, y_train, y_test):
    tree_clf = DecisionTreeClassifier(
        max_depth=6,
        random_state=27
    )
    tree_clf.fit(X_train, y_train)

    preds = tree_clf.predict(X_test)
    probs = tree_clf.predict_proba(X_test)[:, 1]

    metrics = compute_classification_metrics(y_test, preds, probs)
    return tree_clf, metrics