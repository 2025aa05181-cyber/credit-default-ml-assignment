from xgboost import XGBClassifier
from model.metrics_helper import compute_classification_metrics

def train_xgboost(X_train, X_test, y_train, y_test):
    xgb_clf = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=27
    )

    xgb_clf.fit(X_train, y_train)

    preds = xgb_clf.predict(X_test)
    probs = xgb_clf.predict_proba(X_test)[:, 1]

    metrics = compute_classification_metrics(y_test, preds, probs)
    return xgb_clf, metrics