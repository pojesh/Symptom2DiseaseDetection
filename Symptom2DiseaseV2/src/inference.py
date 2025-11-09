
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os
import logging
from dotenv import load_dotenv
from tqdm import tqdm
import torch
import yaml

import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)

# --- Configuration Loading ---
def load_config():
    """Loads the YAML configuration file."""
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

config = load_config()

# --- Constants from Config ---
DB_PATH = config['database']['chroma_db_path']
COLLECTION_NAME = config['database']['collection_name']
MODEL_NAME = 'paraphrase-MiniLM-L6-v2' # This is a fixed model for sentence transformation
SIMILARITY_THRESHOLD = 0.45
BATCH_SIZE = 121
RF_MODEL_PATH = config['model_paths']['rf_model']
LE_MODEL_PATH = config['model_paths']['label_encoder']
SYMPTOM_COLS_PATH = config['model_paths']['symptom_columns']
GEMINI_MODEL = config['generative_model']

# --- Model and Data Loading ---
def load_models_and_data():
    """Loads all necessary models and data files."""
    logging.info("Loading models and data...")
    try:
        clf = joblib.load(RF_MODEL_PATH)
        le = joblib.load(LE_MODEL_PATH)
        symptom_columns = joblib.load(SYMPTOM_COLS_PATH)
        medquad_df = pd.read_csv('data/medquad.csv')
        logging.info("Models and data loaded successfully.")
        return clf, le, symptom_columns, medquad_df
    except FileNotFoundError as e:
        logging.error(f"Error loading models or data: {e}")
        raise

def initialize_sentence_transformer():
    """Initializes the Sentence Transformer model."""
    logging.info("Initializing Sentence Transformer model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(MODEL_NAME, device=device)
    logging.info(f"Sentence Transformer model initialized on {device}.")
    return model

def initialize_chromadb(medquad_df):
    """Initializes ChromaDB and populates it if empty."""
    logging.info("Initializing ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        if collection.count() == 0:
            logging.info("ChromaDB collection is empty. Populating...")
            medquad_df.dropna(subset=['question', 'answer', 'focus_area'], inplace=True)
            documents = [f"Question: {row['question']}\nFocus Area: {row['focus_area']}\n\n{row['answer']}" for _, row in medquad_df.iterrows()]
            ids = [str(i) for i in range(len(documents))]
            for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Adding documents to ChromaDB"):
                collection.add(documents=documents[i:i+BATCH_SIZE], ids=ids[i:i+BATCH_SIZE])
            logging.info("Documents added to ChromaDB successfully.")
        else:
            logging.info("ChromaDB collection already populated.")
        return collection
    except Exception as e:
        logging.error(f"Error initializing ChromaDB: {e}")
        raise

# --- Core Logic ---
def get_top_symptoms(query, symptoms, model, threshold=SIMILARITY_THRESHOLD):
    """Finds the most similar symptoms to the user's query."""
    query_embedding = model.encode([query])
    symptom_embeddings = model.encode(symptoms)
    sim_scores = cosine_similarity(query_embedding, symptom_embeddings)[0]
    return [symptoms[i] for i, score in enumerate(sim_scores) if score >= threshold]

def predict_disease(symptoms, clf, le, symptom_columns):
    """Predicts the top 3 diseases based on the detected symptoms."""
    input_vec = np.zeros(len(symptom_columns), dtype=int)
    for s in symptoms:
        if s in symptom_columns:
            idx = symptom_columns.index(s)
            input_vec[idx] = 1
    probas = clf.predict_proba([input_vec])[0]
    top3_idx = probas.argsort()[-3:][::-1]
    return le.inverse_transform(top3_idx), probas[top3_idx]

def retrieve_evidence(query, collection, disease=None, k=3):
    """Retrieves evidence from ChromaDB."""
    query_text = query
    if disease:
        query_text += " " + disease
    results = collection.query(query_texts=[query_text], n_results=k)
    return results['documents']

def generate_explanation(mode, query, evidence, symptoms=None, diseases=None):
    """Generates a user-friendly explanation using the Gemini API."""
    logging.info(f"Generating explanation with Gemini API in '{mode}' mode...")
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)

        if mode == 'prediction':
            prompt = f"""
                You are a medical guidance assistant. Your task is to explain the predicted disease based on user symptoms and retrieved evidence.

                User Query: {query}
                Extracted Symptoms: {symptoms}
                Predicted Disease(s): {', '.join(diseases)}
                Retrieved Evidence Summary: {evidence}

                Instructions:
                1. Start with a clear warning that this is not a medical diagnosis and the user should consult a doctor.
                2. Provide a concise, user-friendly explanation for each predicted disease, explaining its likelihood based on the given symptoms.
                3. Summarize how the evidence supports the prediction.
                4. Do **not** quote or copy text directly from the evidence; instead, paraphrase key insights.
                5. Your output should be factual, empathetic, and easy for a non-expert user to understand.
            """
        elif mode == 'rag_only':
            prompt = f"""
                You are a medical guidance assistant. You are in a mode where you cannot make a formal prediction. Instead, your task is to analyze the user's query and retrieved medical documents to provide a helpful, general-purpose answer.

                User Query: {query}
                Retrieved Medical Information: {evidence}

                Instructions:
                1. Start with a clear warning that this is not a medical diagnosis and the user should consult a doctor.
                2. Analyze the user's query and the retrieved information.
                3. Based **only** on the retrieved information, identify potential conditions or topics that seem most relevant to the user's query.
                4. For each potential condition, explain what it is and why it might be relevant, citing the user's symptoms.
                5. Do **not** present this as a prediction. Use phrases like "The retrieved information suggests that symptoms like yours could be related to..." or "One possibility mentioned in the medical texts is...".
                6. Your output should be informative, cautious, and guide the user toward professional medical advice.
                7. Use the technical terms and jargon from the medical documents only when necessary. Avoid using jargon that a non-expert user would not understand.
                8. Highlight the disease condition in the beginning itself, for example: "The retrieved information suggests that symptoms like yours could be related to...". 
            """
        else:
             prompt = f"""
                You are a medical guidance assistant. Your primary symptom detection model did not find any relevant symptoms in the user's query. Your task is to provide a helpful response based on the user's raw query.

                User Query: {query}
                Retrieved Medical Information: {evidence}

                Instructions:
                1. Start with a clear warning that this is not a medical diagnosis and the user should consult a doctor.
                2. Acknowledge that you couldn't identify specific symptoms for your prediction model from the query.
                3. Analyze the user's query and the retrieved information to provide general guidance.
                4. Explain what the retrieved information says about the symptoms or situation described by the user.
                5. Do **not** attempt to diagnose. Instead, offer context and suggest what the user might discuss with a healthcare professional.
                6. Frame the response empathetically, for example: "While I couldn't match your description to specific symptoms for a prediction, here is some information based on what you've described...".
                7. Use the technical terms and jargon from the medical documents only when necessary. Avoid using jargon that a non-expert user would not understand.
                8. Highlight the disease condition in the beginning itself, for example: "The retrieved information suggests that symptoms like yours could be related to...". 
            """

        response = model.generate_content(prompt)
        logging.info("Explanation generated successfully.")
        return response.text
    except Exception as e:
        logging.error(f"Error generating explanation with Gemini API: {e}")
        raise

def run_inference(query, models, collection, sentence_model):
    """
    Runs the full inference pipeline based on the configuration.
    Returns a dictionary with results for the UI.
    """
    use_rf = config.get('use_rf_prediction', True)
    clf, le, symptom_columns, _ = models

    if use_rf:
        logging.info("Running in RF prediction mode.")
        top_symptoms = get_top_symptoms(query, symptom_columns, sentence_model)

        if not top_symptoms:
            logging.warning("No relevant symptoms found. Falling back to RAG-only mode.")
            evidence = retrieve_evidence(query, collection)
            explanation = generate_explanation(mode='fallback', query=query, evidence=evidence)
            return {"status": "fallback", "explanation": explanation}

        top3_diseases, top3_probs = predict_disease(top_symptoms, clf, le, symptom_columns)
        evidence = retrieve_evidence(query, collection, disease=top3_diseases[0])
        explanation = generate_explanation(
            mode='prediction',
            query=query,
            symptoms=top_symptoms,
            diseases=top3_diseases,
            evidence=evidence
        )
        return {
            "status": "success",
            "symptoms": top_symptoms,
            "predictions": list(zip(top3_diseases, top3_probs)),
            "explanation": explanation
        }
    else:
        logging.info("Running in RAG-only mode.")
        evidence = retrieve_evidence(query, collection)
        explanation = generate_explanation(mode='rag_only', query=query, evidence=evidence)
        return {"status": "rag_only", "explanation": explanation}
