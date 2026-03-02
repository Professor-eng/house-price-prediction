from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def score(model, X_train, y_train):
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kf,
        scoring="neg_root_mean_squared_error"
    )
    return -scores.mean(), scores.std()

def score_all(models: dict, X_train, y_train, X_holdout, y_holdout):
    results = {}
    for name, model in models.items():
        cv_rmse, cv_std = score(model, X_train, y_train) if name != "stack" else (None, None)
        preds = model.predict(X_holdout)
        holdout_rmse = np.sqrt(mean_squared_error(y_holdout, preds))
        results[name] = {
            "cv_rmse": cv_rmse,
            "cv_std": cv_std,
            "holdout_rmse": holdout_rmse
        }
    return results