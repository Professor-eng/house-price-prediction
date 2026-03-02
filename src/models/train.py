from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.pipeline import Pipeline
from src.config import MODEL_DIR

def load_studies():
    return {
        "lgb": joblib.load(MODEL_DIR/"study_lgb.pkl"),
        "rf": joblib.load(MODEL_DIR/"study_rf.pkl"),
        "ridge": joblib.load(MODEL_DIR/"study_ridge.pkl"),
        "xgb": joblib.load(MODEL_DIR/"study_xgb.pkl"),
    }


def get_base_models():
    studies = load_studies()
    return {
        "lgb": LGBMRegressor(**studies["lgb"].best_params),
        "xgb": XGBRegressor(**studies["xgb"].best_params),
        "rf": RandomForestRegressor(**studies["rf"].best_params),
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(**studies["ridge"].best_params))
        ])
    }

def train_all(X, y):
    models = get_base_models()
    for model in models.values():
        model.fit(X, y)
    return models





def save_model(model, filename):
    path = MODEL_DIR / filename
    joblib.dump(model, path)
