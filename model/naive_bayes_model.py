from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from model.metrics_helper import compute_classification_metrics

def train_naive_bayes(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nb_clf = GaussianNB()
    nb_clf.fit(X_train_scaled, y_train)

    preds = nb_clf.predict(X_test_scaled)
    probs = nb_clf.predict_proba(X_test_scaled)[:, 1]

    metrics = compute_classification_metrics(y_test, preds, probs)
    return (scaler, nb_clf), metrics