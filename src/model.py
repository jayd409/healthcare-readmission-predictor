import numpy as np
import pandas as pd
from ml_utils import normalize, logistic_regression, logit_predict, accuracy, roc_auc_approx

def train_disease_predictor(df):
    """
    Train logistic regression model to predict heart disease risk.
    Returns trained model, metrics, and feature importance.
    """
    # Separate features and target
    X = df.drop('target', axis=1).values
    y = df['target'].values
    feature_names = df.drop('target', axis=1).columns

    # Train/test split (80/20)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))

    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    # Normalize features
    X_train_norm, mean, std = normalize(X_train)
    X_test_norm = (X_test - mean) / std

    # Train logistic regression
    theta = logistic_regression(X_train_norm, y_train, lr=0.1, epochs=1000)

    # Predictions
    proba_train, pred_train = logit_predict(X_train_norm, theta)
    proba_test, pred_test = logit_predict(X_test_norm, theta)

    # Metrics
    train_acc = accuracy(y_train, pred_train)
    test_acc = accuracy(y_test, pred_test)
    test_auc = roc_auc_approx(y_test, proba_test)

    # Feature importance (absolute value of normalized coefficients)
    coef_importance = np.abs(theta[1:])  # Skip bias
    coef_importance = coef_importance / coef_importance.sum()

    # Confusion matrix
    tp = ((pred_test == 1) & (y_test == 1)).sum()
    tn = ((pred_test == 0) & (y_test == 0)).sum()
    fp = ((pred_test == 1) & (y_test == 0)).sum()
    fn = ((pred_test == 0) & (y_test == 1)).sum()

    metrics = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'auc': test_auc,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'sensitivity': round(tp / (tp + fn + 1e-10), 4),
        'specificity': round(tn / (tn + fp + 1e-10), 4),
    }

    # Add risk scores to original dataframe
    df_with_scores = df.copy()
    X_all_norm = (X - mean) / std
    proba_all, _ = logit_predict(X_all_norm, theta)
    df_with_scores['risk_score'] = proba_all

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': coef_importance
    }).sort_values('importance', ascending=False)

    return metrics, df_with_scores, feature_importance, (X_test_norm, y_test, proba_test)
