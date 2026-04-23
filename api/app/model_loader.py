import joblib
import os
from xgboost import XGBClassifier

BASE = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
)

_model  = None
_scaler = None

def get_model():
    global _model
    if _model is None:
        model = XGBClassifier()
        model.load_model(os.path.join(BASE, "best_model_xgb.json"))
        _model = model
    return _model

def get_scaler():
    global _scaler
    if _scaler is None:
        _scaler = joblib.load(os.path.join(BASE, "scaler.pkl"))
    return _scaler
