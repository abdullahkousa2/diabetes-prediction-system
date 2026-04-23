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
├── notebooks/
│   ├── 01_data_preprocessing.ipynb   ← Data cleaning, feature engineering, SMOTE
│   ├── 02_model_training.ipynb       ← Train RF, XGBoost, MLP, Soft Ensemble + SHAP
│   └── 03_gradio_interface.ipynb     ← Interactive Gradio UI for local testing
├── dataset/
│   ├── diabetes_merged.csv           ← Merged Pima dataset
│   └── plots/                        ← Confusion matrices, ROC curves, SHAP plots
├── api/                              ← Flask REST API + Docker deployment
│   ├── app/                          ← Routes, model loader, utilities
│   ├── models/                       ← Trained XGBoost model + scaler
│   ├── diabetes_interface.html       ← Browser-based test UI
│   ├── Dockerfile
│   └── docker-compose.yml
└── docs/
    └── diabetes_literature_review.pdf
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
- Saves the best model as `best_model_xgb.pkl`

### 3. REST API (`api/`)
- Flask-based API exposing three prediction endpoints
- Loads trained XGBoost model and scaler at startup
- Supports single, explained, and batch predictions
- Includes a browser-based test UI (`diabetes_interface.html`)
- CORS enabled — ready for frontend integration

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### 1. Run the Training Pipeline

```bash
# Install Jupyter and dependencies
pip install jupyter numpy pandas scikit-learn xgboost imbalanced-learn shap matplotlib joblib

# Open and run notebooks in order
jupyter notebook notebooks/01_data_preprocessing.ipynb
jupyter notebook notebooks/02_model_training.ipynb
```

Outputs will be saved to the `dataset/` folder.

### 2. Run the API (Local)

```bash
cd api

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python run.py
```

API available at: `http://localhost:5000`

### 3. Run the API (Docker)

```bash
cd api
docker-compose up --build
```

API available at: `http://localhost:5000`

---

## API Endpoints

| Method | Endpoint           | Description                                      |
|--------|--------------------|--------------------------------------------------|
| GET    | `/health`          | Server and model status                          |
| POST   | `/predict`         | Single patient prediction                        |
| POST   | `/predict/explain` | Prediction with SHAP feature contributions       |
| POST   | `/predict/batch`   | Batch prediction for up to 100 patients          |

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
  "input": { "..." : "..." }
}
```

See [api/README.md](api/README.md) for full endpoint documentation and field ranges.

---

## Dataset

**Pima Indians Diabetes Dataset** — merged from two sources:
- 8 original clinical features
- 4 engineered interaction features
- Binary target: `0` = No Diabetes, `1` = Diabetes
- Class imbalance handled with **SMOTE**

| Feature                      | Description                          |
|------------------------------|--------------------------------------|
| Pregnancies                  | Number of pregnancies                |
| Glucose                      | Plasma glucose concentration         |
| BloodPressure                | Diastolic blood pressure (mm Hg)     |
| SkinThickness                | Triceps skinfold thickness (mm)      |
| Insulin                      | 2-hour serum insulin (µU/ml)         |
| BMI                          | Body mass index                      |
| DiabetesPedigreeFunction     | Diabetes pedigree function score     |
| Age                          | Age in years                         |

---

## Model Results

Four models were trained and compared:

| Model               | Threshold |
|---------------------|-----------|
| Random Forest       | 0.40      |
| XGBoost *(best)*    | 0.40      |
| MLP                 | 0.40      |
| Soft Voting Ensemble| 0.40      |

Evaluation plots are saved in `dataset/`:
- `roc_all_models.png` — ROC curve comparison
- `cm_XGBoost.png`, `cm_Random_Forest.png`, `cm_MLP.png`, `cm_Soft_Ensemble.png`

---

## SHAP Explainability

Global and per-patient explanations generated with `shap.TreeExplainer`:

| Plot                    | File                        | Description                            |
|-------------------------|-----------------------------|----------------------------------------|
| Feature importance      | `shap_importance.png`       | Global bar chart of feature impact     |
| Beeswarm                | `shap_beeswarm.png`         | Direction and magnitude per sample     |
| Patient explanation     | `shap_patient_0.png`        | Force plot for individual patient      |

The API's `/predict/explain` endpoint returns live SHAP contributions per request.

---

## Risk Levels

| Probability | Risk Level |
|-------------|------------|
| ≥ 70%       | High       |
| 40% – 69%   | Moderate   |
| < 40%       | Low        |

---

## Tech Stack

| Component         | Technology              |
|-------------------|-------------------------|
| Language          | Python 3.10             |
| ML Models         | XGBoost, Random Forest, MLP (scikit-learn) |
| Imbalance Handling| imbalanced-learn (SMOTE)|
| Explainability    | SHAP                    |
| API Framework     | Flask 3.0               |
| Containerisation  | Docker + Docker Compose |

---

## License

This project is licensed under the MIT License.
