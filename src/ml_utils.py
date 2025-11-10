import numpy as np

def normalize(X):
    """Standardize features to zero mean and unit variance."""
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std

def linear_regression(X, y):
    """Fit linear regression using normal equations (OLS)."""
    Xb = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(Xb, y, rcond=None)[0]

def lr_predict(X, theta):
    """Predict using linear regression coefficients."""
    Xb = np.column_stack([np.ones(len(X)), X])
    return Xb @ theta

def sigmoid(z):
    """Sigmoid activation function with clipping to prevent overflow."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def logistic_regression(X, y, lr=0.1, epochs=500):
    """Fit logistic regression via gradient descent."""
    Xb = np.column_stack([np.ones(len(X)), X])
    theta = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        h = sigmoid(Xb @ theta)
        grad = Xb.T @ (h - y) / len(y)
        theta -= lr * grad
    return theta

def logit_predict(X, theta, threshold=0.5):
    """Get probabilities and binary predictions from logistic model."""
    Xb = np.column_stack([np.ones(len(X)), X])
    proba = sigmoid(Xb @ theta)
    return proba, (proba >= threshold).astype(int)

def kmeans(X, k, n_iter=100, seed=42):
    """K-means clustering."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), k, replace=False)
    centroids = X[idx].copy().astype(float)
    labels = np.zeros(len(X), dtype=int)
    for _ in range(n_iter):
        dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            mask = labels == i
            if mask.any():
                centroids[i] = X[mask].mean(axis=0)
    return labels, centroids

def r_squared(y_true, y_pred):
    """Compute R-squared metric."""
    ss_res = ((y_true - y_pred)**2).sum()
    ss_tot = ((y_true - y_true.mean())**2).sum()
    return round(1 - ss_res / (ss_tot + 1e-10), 4)

def rmse(y_true, y_pred):
    """Compute root mean squared error."""
    return round(np.sqrt(((y_true - y_pred)**2).mean()), 2)

def accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    return round((y_true == y_pred).mean(), 4)

def roc_auc_approx(y_true, proba):
    """Approximate ROC-AUC using sorted thresholds."""
    thresholds = np.linspace(0, 1, 50)
    tprs, fprs = [], []
    pos = y_true.sum()
    neg = len(y_true) - pos
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        tprs.append(tp / max(pos, 1))
        fprs.append(fp / max(neg, 1))
    tprs, fprs = np.array(tprs), np.array(fprs)
    idx = np.argsort(fprs)
    return round(float(np.trapz(tprs[idx], fprs[idx])), 4)
