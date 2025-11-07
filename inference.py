import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Load model and helpers
clf = joblib.load('rf_model.joblib')
le = joblib.load('label_encoder.joblib')
symptom_columns = joblib.load('symptom_columns.joblib')

# Terminal loop
print("Enter your symptoms as comma-separated list (e.g., itching, joint_pain):")
user_input = input().lower().replace(" ", "").split(',')

# Vectorize input
input_vector = np.zeros(len(symptom_columns))
input_symptoms = set(user_input)
column_indices = {symptom: idx for idx, symptom in enumerate(symptom_columns)}
for symptom in input_symptoms:
    if symptom in column_indices:
        input_vector[column_indices[symptom]] = 1

# Predict
probas = clf.predict_proba([input_vector])[0]
top3_idx = probas.argsort()[-3:][::-1]
top3_diseases = le.inverse_transform(top3_idx)
top3_probs = probas[top3_idx]

print("\nTop likely diseases:")
for disease, prob in zip(top3_diseases, top3_probs):
    print(f"{disease}: {prob*100:.2f}% likelihood")
