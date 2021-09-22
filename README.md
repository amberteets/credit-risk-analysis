# Credit Risk Analysis with Supervised Machine Learning

## Purpose

Investigate the efficacy of various machine learning sampling techniques and algorithms for predicting the credit risk of credit card applicants.

## Approach

### Using Resampling Models to Predict Credit Risk

With knowledge of `imbalanced-learn` and `scikit-learn` libraries, evaluate three machine learning models by using resampling to determine which is better at predicting credit risk.

Oversampling Algorithms: `RandomOverSampler`, `SMOTE`.

Undersampling Algorithm: `ClusterCentroids`.

Combination Algorithm: `SMOTEENN`

For each algorithm:

- Resample dataset
- View count of target classes
- Train logistic regression classifier
- Calculate balanced accuracy score
- Generate confusion matrix
- Generate classification report