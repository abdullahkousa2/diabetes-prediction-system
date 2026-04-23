import numpy as np

FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
    'Glucose_BMI', 'Age_BMI', 'Glucose_squared', 'Insulin_BMI'
]

FIELDS = {
    'pregnancies':                {'min': 0,    'max': 17},
    'glucose':                    {'min': 50,   'max': 250},
    'blood_pressure':             {'min': 40,   'max': 130},
    'skin_thickness':             {'min': 0,    'max': 100},
    'insulin':                    {'min': 0,    'max': 900},
    'bmi':                        {'min': 10.0, 'max': 70.0},
    'diabetes_pedigree_function': {'min': 0.0,  'max': 2.5},
    'age':                        {'min': 18,   'max': 90},
}

def validate(data):
    cleaned = {}
    for field, rules in FIELDS.items():
        if field not in data:
            return f"Missing required field: '{field}'", None
        try:
            val = float(data[field])
        except (ValueError, TypeError):
            return f"Field '{field}' must be a number", None
        if val < rules['min'] or val > rules['max']:
            return f"Field '{field}' must be between {rules['min']} and {rules['max']}", None
        cleaned[field] = val
    return None, cleaned

def build_feature_vector(cleaned):
    g   = cleaned['glucose']
    bmi = cleaned['bmi']
    age = cleaned['age']
    ins = cleaned['insulin']

    return np.array([[
        cleaned['pregnancies'],
        g,
        cleaned['blood_pressure'],
        cleaned['skin_thickness'],
        ins,
        bmi,
        cleaned['diabetes_pedigree_function'],
        age,
        g   * bmi,
        age * bmi,
        g   ** 2,
        ins * bmi,
    ]])
