from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from src.models.train import get_base_models
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_stack():
    models = get_base_models()
    return StackingRegressor(
        estimators=list(models.items()),
        final_estimator=Pipeline([("scaler", StandardScaler()),("model", Ridge())]),
        cv=5
    )

def train_stack(X_train, y_train):
    stack = build_stack()
    stack.fit(X_train, y_train)
    return stack