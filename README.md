# Credit Card Default Prediction - ML Assignment 2

This repository contains implementation of multiple classification models
and a Streamlit application as part of BITS M.Tech ML Assignment 2.


| ML Model                 | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | -------- | ------ |
| Logistic Regression      | 0.8104   | 0.7264 | 0.7083    | 0.2429 | 0.3618   | 0.3362 |
| Decision Tree            | 0.8191   | 0.7562 | 0.6674    | 0.3629 | 0.4701   | 0.3975 |
| KNN                      | 0.8025   | 0.7163 | 0.5935    | 0.3406 | 0.4328   | 0.3420 |
| Naive Bayes              | 0.7549   | 0.7417 | 0.4595    | 0.6118 | 0.5248   | 0.3710 |
| Random Forest (Ensemble) | 0.8205   | 0.7821 | 0.6768    | 0.3611 | 0.4709   | 0.4015 |
| XGBoost (Ensemble)       | 0.8175   | 0.7867 | 0.6608    | 0.3593 | 0.4654   | 0.3916 |


| ML Model Name            | Observation about Model Performance                                                                                                                                                                                                                                                 |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Logistic Regression achieved good overall accuracy but exhibited very low recall, indicating that it failed to correctly identify a large proportion of defaulters. This suggests that the relationship between features and the target variable is not strictly linear in this dataset. |
| Decision Tree            | The Decision Tree model showed improved recall and F1-score compared to Logistic Regression by capturing non-linear patterns in the data. However, its performance is limited by potential overfitting, which restricts further generalization.                                          |
| kNN                      | The kNN model demonstrated moderate performance across all metrics. Its reliance on distance-based calculations in a high-dimensional feature space likely reduced its effectiveness, resulting in a lower AUC compared to ensemble models.                                              |
| Naive Bayes              | Naive Bayes achieved the highest recall among all models, indicating strong ability to identify defaulters. However, its relatively low precision reflects the violation of the conditional independence assumption among features in this dataset.                                      |
| Random Forest (Ensemble) | Random Forest delivered the most balanced performance across accuracy, AUC, F1-score, and MCC. By combining multiple decision trees, it reduced variance and effectively captured complex feature interactions.                                                                          |
| XGBoost (Ensemble)       | XGBoost achieved the highest AUC score, demonstrating superior ranking capability between defaulters and non-defaulters. Its gradient boosting approach iteratively corrected errors, resulting in strong and robust predictive performance.                                             |
