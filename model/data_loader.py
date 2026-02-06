import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(csv_path):
    raw_df = pd.read_csv(csv_path)
    target_column = "default.payment.next.month"

    feature_matrix = raw_df.drop(columns=[target_column])
    target_vector = raw_df[target_column]

    features_train, features_test, labels_train, labels_test = train_test_split(
        feature_matrix,
        target_vector,
        test_size=0.25,
        random_state=27,
        stratify=target_vector
    )
    return features_train, features_test, labels_train, labels_test