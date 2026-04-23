from flask import Blueprint, jsonify
from ..model_loader import get_model, get_scaler

health_bp = Blueprint("health", __name__)

@health_bp.route("/health", methods=["GET"])
def health():
    try:
        get_model()
        get_scaler()
        status = "ok"
        message = "Model and scaler loaded successfully"
    except Exception as e:
        status = "error"
        message = str(e)

    return jsonify({
        "status":  status,
        "message": message,
        "version": "1.0.0",
        "model":   "XGBoost"
    }), 200 if status == "ok" else 500
