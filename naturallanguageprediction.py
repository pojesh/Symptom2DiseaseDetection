from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")


symptoms = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 
    'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 
    'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 
    'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 
    'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 
    'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 
    'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 
    'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 
    'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 
    'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 
    'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 
    'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 
    'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 
    'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 
    'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 
    'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 
    'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 
    'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 
    'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 
    'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 
    'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 
    'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 
    'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 
    'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 
    'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 
    'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
]

symptom_embeddings = model.encode(symptoms)

#user_input = "My nose is runny and I can't stop sneezing"
user_input = "My chest is paining and im unable to breathe" 

user_embedding = model.encode([user_input])[0].reshape(1, -1)

similarities = cosine_similarity(user_embedding, symptom_embeddings)[0]
top_k = 5
best_indices = similarities.argsort()[-top_k:][::-1]
symptom_similarities = []
for idx in best_indices:
    symptom_similarities.append((symptoms[idx],float(f"{similarities[idx]:.3f}")))
    print(f"{symptoms[idx]} -> Similarity: {similarities[idx]:.3f}")


print(symptom_similarities)


import warnings
warnings.filterwarnings("ignore")


import joblib
import pandas as pd

SIMILARITY_THRESHOLD = 0.45
selected_symptoms = [symptom for symptom, sim in symptom_similarities if sim >= SIMILARITY_THRESHOLD]

# Load your feature column order (same as you saved during training)
symptom_columns = joblib.load('symptom_columns.joblib')

# Prepare the binary model input vector
input_vector = np.zeros(len(symptom_columns), dtype=int)
for symptom in selected_symptoms:
    if symptom in symptom_columns:
        idx = symptom_columns.index(symptom)
        input_vector[idx] = 1

# Wrap into DataFrame for model
input_df = pd.DataFrame([input_vector], columns=symptom_columns)

# Random Forest inference
clf = joblib.load('rf_model.joblib')
le = joblib.load('label_encoder.joblib')
probas = clf.predict_proba(input_df)[0]
top3_idx = probas.argsort()[-3:][::-1]
top3_diseases = le.inverse_transform(top3_idx)
top3_probs = probas[top3_idx]

for disease, prob in zip(top3_diseases, top3_probs):
    print(f"{disease}: {prob*100:.2f}% likelihood")