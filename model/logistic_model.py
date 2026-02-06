from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from model.metrics_helper import compute_classification_metrics

def train_logistic_regression(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1200))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    metrics = compute_classification_metrics(y_test, preds, probs)
    return pipeline, metrics