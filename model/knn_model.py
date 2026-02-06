from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from model.metrics_helper import compute_classification_metrics

def train_knn(X_train, X_test, y_train, y_test):
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=7))
    ])

    knn_pipeline.fit(X_train, y_train)
    preds = knn_pipeline.predict(X_test)
    probs = knn_pipeline.predict_proba(X_test)[:, 1]

    metrics = compute_classification_metrics(y_test, preds, probs)
    return knn_pipeline, metrics