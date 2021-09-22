# Credit Risk Analysis with Supervised Machine Learning

## Purpose

Investigate the efficacy of various machine learning sampling techniques and algorithms for predicting the credit risk of credit card applicants.

## Approach

### Data Cleaning

1. Select feature and target columns from DataFrame.
    - For features, drop identification columns (e.g. 'member_id')
    - Target is 'loan_status'
2. Drop completely null columns, and rows with any null values.
3. Drop 'Issued' loan status (not enough information to map as low- or high-risk).
4. Convert interest rate from string to numerical data type.
5. Convert target column values to 'low-risk' or 'high-risk' based on values.
    - 'Current' loan status denotes low-risk loans.
    - 'Late', 'Default', and 'Grace Period' denote high-risk loans.

### Resampling Models to Predict Credit Risk

With `imbalanced-learn` and `scikit-learn` libraries, evaluate four machine learning models by using resampling to determine which is better at predicting credit risk.

- Oversampling Algorithms: `RandomOverSampler`, `SMOTE`.
- Undersampling Algorithm: `ClusterCentroids`.
- Combination Algorithm: `SMOTEENN`

For each algorithm:

- Resample dataset
- View count of target classes
- Train logistic regression classifier
- Calculate balanced accuracy score
- Generate confusion matrix
- Generate classification report

### Ensemble Classifiers to Predict Credit Risk

With `imbalanced-learn.ensemble` library, evaluate two different ensemble classifiers - `BalancedRandomForestClassifier` and `EasyEnsembleClassifier` - to predict credit risk.

For each algorithm:

- Resample dataset
- View count of target classes
- Train logistic regression classifier
- Calculate balanced accuracy score
- Generate confusion matrix
- Generate classification report