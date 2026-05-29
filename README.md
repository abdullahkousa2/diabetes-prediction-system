# Diabetes Prediction System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-blueviolet)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end machine learning system for diabetes risk prediction. Covers the full pipeline from data preprocessing and model training through to a production-ready REST API with SHAP explainability and Docker deployment.

---

## Demo

> Enter patient vitals → get a risk prediction (High / Moderate / Low) + SHAP breakdown showing which features pushed the score up or down.


https://github.com/user-attachments/assets/f7baf55c-303c-40a6-92fc-2186de7aedc0


<!-- Add a screenshot or GIF here: drag your demo image into the GitHub editor and it will upload automatically -->
<!-- Example: ![Diabetes Demo](assets/demo.gif) -->

---

## Architecture

```
diabetes-prediction-system/
├── app/                              ← Flask application
│   ├── __init__.py                   ← App factory + CORS
│   ├── model_loader.py               ← XGBoost model & scaler loader
│   ├── utils.py                      ← Validation + feature engineering
│   └── routes/
│       ├── predict.py                ← /predict, /predict/explain, /predict/batch
│       └── health.py                 ← /health
├── models/
│   ├── best_model_xgb.json           ← Trained XGBoost model
│   └── scaler.pkl                    ← Feature scaler
├── notebooks/
│   ├── 01_data_preprocessing.ipynb   ← Cleaning, feature engineering, SMOTE
│   ├── 02_model_training.ipynb       ← Train RF, XGBoost, MLP, Ensemble + SHAP
│   └── 03_gradio_interface.ipynb     ← Interactive Gradio UI for local testing
├── dataset/
│   ├── diabetes_merged.csv           ← Merged Pima dataset
│   └── plots/                        ← Confusion matrices, ROC curves, SHAP plots
├── static/
│   └── interface.html                ← Browser-based test UI
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run.py                            ← API entry point
```

---

## Model Results

All models evaluated at threshold **0.40** (tuned to maximise recall on the diabetic class — catching missed cases matters more than false alarms in a medical context).

| Model | Accuracy | ROC-AUC |
|---|---|---|
| XGBoost *(best, deployed)* | **93%** | **0.9896** |
| Random Forest | 90% | 0.9810 |
| Soft Voting Ensemble | 89% | 0.9824 |
| MLP | 83% | 0.9298 |

---

## Components

### 1. Preprocessing Pipeline (`notebooks/01_data_preprocessing.ipynb`)
- Merges two Pima Indians Diabetes dataset sources
- Handles missing values and outliers
- Engineers interaction features: `Glucose_BMI`, `Age_BMI`, `Glucose_squared`, `Insulin_BMI`
- Applies **SMOTE** to balance the minority class
- Outputs: `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`

### 2. Model Training (`notebooks/02_model_training.ipynb`)
- Trains and evaluates four models: **Random Forest**, **XGBoost**, **MLP**, **Soft Voting Ensemble**
- Decision threshold tuned to **0.40** to optimise recall for the diabetic class
- Generates confusion matrices and ROC curves for all models
- Runs **SHAP TreeExplainer** for global and per-patient feature importance
- Saves the best model as `models/best_model_xgb.json`

### 3. REST API (`app/` + `run.py`)
- Flask-based API exposing three prediction endpoints
- Loads trained XGBoost model and scaler at startup
- Supports single, explained, and batch predictions
- CORS enabled — ready for frontend integration

---

## Quick Start

### Prerequisites
- Python 3.10+

### 1. Run the Training Pipeline

```bash
pip install jupyter numpy pandas scikit-learn xgboost imbalanced-learn shap matplotlib joblib
jupyter notebook notebooks/01_data_preprocessing.ipynb
jupyter notebook notebooks/02_model_training.ipynb
```

### 2. Run the API (Local)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

API available at: `http://localhost:5000`

Open `static/interface.html` in your browser for the interactive test UI.

### 3. Run the API (Docker)

```bash
docker-compose up --build
```

API available at: `http://localhost:5000`

---

## API Endpoints

| Method | Endpoint             | Description                                      |
|--------|----------------------|--------------------------------------------------|
| GET    | `/health`            | Server and model status                          |
| POST   | `/predict`           | Single patient prediction                        |
| POST   | `/predict/explain`   | Prediction with SHAP feature contributions       |
| POST   | `/predict/batch`     | Batch prediction for up to 100 patients          |

### Example Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 2,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 150,
    "bmi": 33.6,
    "diabetes_pedigree_function": 0.627,
    "age": 50
  }'
```

### Example Response

```json
{
  "prediction": "Diabetic",
  "probability": 0.8231,
  "risk_level": "High",
  "threshold": 0.4,
  "input": { "...": "..." }
}
```

---

## SHAP Explainability

Global and per-patient explanations via `shap.TreeExplainer`:

| Plot | Description |
|---|---|
| Feature importance | Global bar chart — which features matter most across all patients |
| Beeswarm | Direction and magnitude of each feature's impact per sample |
| Patient explanation | Force plot for a single patient — why this specific score |

The `/predict/explain` endpoint returns live SHAP contributions per request, with plain-language explanations in the browser UI.

---

## Dataset

**Pima Indians Diabetes Dataset** — merged from two sources:

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-hour serum insulin (µU/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes pedigree function score |
| Age | Age in years |

---

## Risk Levels

| Probability | Risk Level |
|-------------|------------|
| ≥ 70% | High |
| 40% – 69% | Moderate |
| < 40% | Low |

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10 |
| ML Models | XGBoost, Random Forest, MLP (scikit-learn) |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| API Framework | Flask 3.0 |
| Containerisation | Docker + Docker Compose |

---

## License

MIT License — see [LICENSE](LICENSE).


https://github.com/user-attachments/assets/8b10356a-f8e4-401c-9b47-1d13a2c710ec



https://github.com/user-attachments/assets/24da2b45-bdec-4413-b946-8964462ce745

