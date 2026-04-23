# Diabetes Prediction System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-blueviolet)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end machine learning system for diabetes risk prediction. Covers the full pipeline from data preprocessing and model training through to a production-ready REST API with SHAP explainability and Docker deployment.

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
├── docs/
│   └── diabetes_literature_review.pdf
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run.py                            ← API entry point
```

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

## Dataset

**Pima Indians Diabetes Dataset** — merged from two sources:

| Feature                  | Description                          |
|--------------------------|--------------------------------------|
| Pregnancies              | Number of pregnancies                |
| Glucose                  | Plasma glucose concentration         |
| BloodPressure            | Diastolic blood pressure (mm Hg)     |
| SkinThickness            | Triceps skinfold thickness (mm)      |
| Insulin                  | 2-hour serum insulin (µU/ml)         |
| BMI                      | Body mass index                      |
| DiabetesPedigreeFunction | Diabetes pedigree function score     |
| Age                      | Age in years                         |

---

## Model Results

Four models trained and compared — all with threshold **0.40** to maximise recall on the diabetic class:

| Model                | Accuracy |
|----------------------|----------|
| Random Forest        | —        |
| XGBoost *(best)*     | —        |
| MLP                  | —        |
| Soft Voting Ensemble | —        |

Evaluation plots saved in `dataset/plots/`:
- `roc_all_models.png` — ROC curve comparison
- `cm_XGBoost.png`, `cm_Random_Forest.png`, `cm_MLP.png`, `cm_Soft_Ensemble.png`

---

## SHAP Explainability

Global and per-patient explanations via `shap.TreeExplainer`:

| Plot                | File                  | Description                        |
|---------------------|-----------------------|------------------------------------|
| Feature importance  | `shap_importance.png` | Global bar chart of feature impact |
| Beeswarm            | `shap_beeswarm.png`   | Direction and magnitude per sample |
| Patient explanation | `shap_patient_0.png`  | Force plot for individual patient  |

The `/predict/explain` endpoint returns live SHAP contributions per request, with plain-language explanations in the browser UI.

---

## Risk Levels

| Probability | Risk Level |
|-------------|------------|
| ≥ 70%       | High       |
| 40% – 69%   | Moderate   |
| < 40%       | Low        |

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

## Tech Stack

| Component          | Technology                              |
|--------------------|-----------------------------------------|
| Language           | Python 3.10                             |
| ML Models          | XGBoost, Random Forest, MLP (scikit-learn) |
| Imbalance Handling | imbalanced-learn (SMOTE)                |
| Explainability     | SHAP                                    |
| API Framework      | Flask 3.0                               |
| Containerisation   | Docker + Docker Compose                 |

---

## License

This project is licensed under the MIT License.
