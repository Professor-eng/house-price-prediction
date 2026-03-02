import joblib
import numpy as np
from src.config import MODEL_DIR

def load_model(filename):
    path = MODEL_DIR / filename
    return joblib.load(path)

def predict(model, X):
    log_predictions = model.predict(X)
    return np.expm1(log_predictions) 