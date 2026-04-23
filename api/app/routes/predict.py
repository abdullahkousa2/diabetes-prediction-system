from flask import Blueprint, request, jsonify
import shap
import numpy as np

from ..model_loader  import get_model, get_scaler
from ..utils         import validate, build_feature_vector, FEATURE_NAMES

predict_bp = Blueprint("predict", __name__)

THRESHOLD = 0.40

def _risk_level(proba):
    if proba >= 0.70:  return "High"
    if proba >= 0.40:  return "Moderate"
    return "Low"

def _get_expected_value(ev):
    if hasattr(ev, '__len__'):
        return float(ev[0])
    return float(ev)


# ── POST /predict ────────────────────────────────────────────────
@predict_bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    error, cleaned = validate(data)
    if error:
        return jsonify({"error": error}), 422

    model  = get_model()
    scaler = get_scaler()
    raw    = build_feature_vector(cleaned)
    scaled = scaler.transform(raw)
    proba  = float(model.predict_proba(scaled)[0][1])
    label  = "Diabetic" if proba >= THRESHOLD else "Non-Diabetic"

    return jsonify({
        "prediction":  label,
        "probability": round(proba, 4),
        "risk_level":  _risk_level(proba),
        "threshold":   THRESHOLD,
        "input":       cleaned
    }), 200


# ── POST /predict/explain ────────────────────────────────────────
@predict_bp.route("/predict/explain", methods=["POST"])
def predict_explain():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    error, cleaned = validate(data)
    if error:
        return jsonify({"error": error}), 422

    model  = get_model()
    scaler = get_scaler()
    raw    = build_feature_vector(cleaned)
    scaled = scaler.transform(raw)
    proba  = float(model.predict_proba(scaled)[0][1])
    label  = "Diabetic" if proba >= THRESHOLD else "Non-Diabetic"

    booster = model.get_booster()
    booster.set_param("base_score", 0.5)
    explainer   = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(scaled)[0]
    base_rate   = _get_expected_value(explainer.expected_value)
    contributions = sorted([
        {
            "feature":    FEATURE_NAMES[i],
            "shap_value": round(float(shap_values[i]), 4),
            "direction":  "increases risk" if shap_values[i] > 0 else "decreases risk"
        }
        for i in range(len(FEATURE_NAMES))
    ], key=lambda x: abs(x["shap_value"]), reverse=True)

    return jsonify({
        "prediction":    label,
        "probability":   round(proba, 4),
        "risk_level":    _risk_level(proba),
        "threshold":     THRESHOLD,
        "base_rate":     round(base_rate, 4),
        "contributions": contributions,
        "input":         cleaned
    }), 200


# ── POST /predict/batch ──────────────────────────────────────────
@predict_bp.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True)
    if not data or "patients" not in data:
        return jsonify({"error": "Body must contain a 'patients' list"}), 400

    patients = data["patients"]
    if not isinstance(patients, list) or len(patients) == 0:
        return jsonify({"error": "'patients' must be a non-empty list"}), 400

    if len(patients) > 100:
        return jsonify({"error": "Batch size cannot exceed 100 patients"}), 400

    model  = get_model()
    scaler = get_scaler()
    results = []

    for i, patient in enumerate(patients):
        error, cleaned = validate(patient)
        if error:
            results.append({"patient_index": i, "error": error})
            continue

        raw    = build_feature_vector(cleaned)
        scaled = scaler.transform(raw)
        proba  = float(model.predict_proba(scaled)[0][1])
        label  = "Diabetic" if proba >= THRESHOLD else "Non-Diabetic"

        results.append({
            "patient_index": i,
            "prediction":    label,
            "probability":   round(proba, 4),
            "risk_level":    _risk_level(proba),
        })

    return jsonify({"total": len(patients), "results": results}), 200
