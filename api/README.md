# Diabetes Prediction API

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

A REST API for diabetes risk prediction powered by an XGBoost classifier with SHAP explainability. Supports single-patient, explained, and batch predictions.

---

## Project Structure

```
api/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── model_loader.py      # Loads XGBoost model and scaler
│   ├── utils.py             # Input validation and feature engineering
│   └── routes/
│       ├── predict.py       # /predict, /predict/explain, /predict/batch
│       └── health.py        # /health
├── models/
│   ├── best_model_xgb.json  # Trained XGBoost model
│   └── scaler.pkl           # Feature scaler
├── diabetes_interface.html  # Browser-based test UI
├── run.py                   # App entry point
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Getting Started

### Local Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
python run.py
```

Server runs at: `http://localhost:5000`

### Docker Setup

```bash
docker-compose up --build
```

Server runs at: `http://localhost:5000`

---

## Browser UI

Open `diabetes_interface.html` in your browser while the server is running for an interactive test interface with live SHAP visualisation.

---

## API Endpoints

| Method | Endpoint             | Description                                |
|--------|----------------------|--------------------------------------------|
| GET    | `/health`            | Server and model status                    |
| POST   | `/predict`           | Single patient prediction                  |
| POST   | `/predict/explain`   | Prediction + SHAP feature contributions    |
| POST   | `/predict/batch`     | Batch prediction (up to 100 patients)      |

### GET `/health`

```json
{
  "status": "ok",
  "message": "Model and scaler loaded successfully",
  "version": "1.0.0",
  "model": "XGBoost"
}
```

### POST `/predict`

**Request:**
```json
{
  "pregnancies": 2,
  "glucose": 148,
  "blood_pressure": 72,
  "skin_thickness": 35,
  "insulin": 150,
  "bmi": 33.6,
  "diabetes_pedigree_function": 0.627,
  "age": 50
}
```

**Response:**
```json
{
  "prediction": "Diabetic",
  "probability": 0.9389,
  "risk_level": "High",
  "threshold": 0.4,
  "input": { "..." : "..." }
}
```

### POST `/predict/explain`

Same request body as `/predict`. Adds SHAP feature contributions:

```json
{
  "prediction": "Diabetic",
  "probability": 0.9389,
  "risk_level": "High",
  "threshold": 0.4,
  "base_rate": -0.0258,
  "contributions": [
    { "feature": "Age",        "shap_value": 0.5413, "direction": "increases risk" },
    { "feature": "Glucose_BMI","shap_value": 0.5392, "direction": "increases risk" }
  ],
  "input": { "..." : "..." }
}
```

### POST `/predict/batch`

```json
{
  "patients": [
    { "pregnancies": 2, "glucose": 148, "blood_pressure": 72, "skin_thickness": 35, "insulin": 150, "bmi": 33.6, "diabetes_pedigree_function": 0.627, "age": 50 },
    { "pregnancies": 0, "glucose": 95,  "blood_pressure": 70, "skin_thickness": 20, "insulin": 80,  "bmi": 25.1, "diabetes_pedigree_function": 0.210, "age": 28 }
  ]
}
```

---

## Input Field Ranges

| Field                        | Type  | Min  | Max  |
|------------------------------|-------|------|------|
| `pregnancies`                | int   | 0    | 17   |
| `glucose`                    | float | 50   | 250  |
| `blood_pressure`             | float | 40   | 130  |
| `skin_thickness`             | float | 0    | 100  |
| `insulin`                    | float | 0    | 900  |
| `bmi`                        | float | 10.0 | 70.0 |
| `diabetes_pedigree_function` | float | 0.0  | 2.5  |
| `age`                        | int   | 18   | 90   |

---

## Risk Levels

| Probability | Risk Level |
|-------------|------------|
| ≥ 70%       | High       |
| 40% – 69%   | Moderate   |
| < 40%       | Low        |
