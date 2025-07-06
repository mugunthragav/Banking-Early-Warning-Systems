import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def featImpMDI(fit, featNames):
    return pd.Series(fit.feature_importances_, index=featNames).sort_values(ascending=False)

def featImpMDA(clf, X, y, cv=None, t1=None, scoring='accuracy', pctEmbargo=0.01):
    if cv is None:
        cv = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)
    imp = []
    for train_idx, test_idx in cv.split(X):
        fit = clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        acc_full = fit.score(X.iloc[test_idx], y.iloc[test_idx])
        imp.append([acc_full] + [fit.score(X.iloc[test_idx].drop(col, axis=1), y.iloc[test_idx])
                                for col in X.columns])
    imp = np.mean(imp, axis=0)
    return pd.DataFrame({'mean': imp[1:] - imp[0], 'std': np.std(imp[1:], axis=0)}, index=X.columns)