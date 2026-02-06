from sklearn.ensemble import RandomForestClassifier
from model.metrics_helper import compute_classification_metrics

def train_random_forest(X_train, X_test, y_train, y_test):
    rf_clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        random_state=27,
        n_jobs=-1
    )
    rf_clf.fit(X_train, y_train)

    preds = rf_clf.predict(X_test)
    probs = rf_clf.predict_proba(X_test)[:, 1]

    metrics = compute_classification_metrics(y_test, preds, probs)
    return rf_clf, metrics