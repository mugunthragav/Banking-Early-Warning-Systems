from sklearn.model_selection import KFold, GridSearchCV
import numpy as np

def cvScore(clf, X, y, scoring='accuracy', cv=None, t1=None, pctEmbargo=0.01):
    if cv is None:
        cv = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in cv.split(X):
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(clf.score(X.iloc[test_idx], y.iloc[test_idx]))
    return np.array(scores)