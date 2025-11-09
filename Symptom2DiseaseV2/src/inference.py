
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

# Constants
DB_PATH = "data/db"
COLLECTION_NAME = "medquad_collection_v4"
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
SIMILARITY_THRESHOLD = 0.45
BATCH_SIZE = 121

def load_models():
    """Loads the machine learning models and data."""
    logging.info("Loading models and data...")
    try:
        clf = joblib.load('models/rf_model.joblib')
        le = joblib.load('models/label_encoder.joblib')
        symptom_columns = joblib.load('models/symptom_columns.joblib')
        symptom_df = pd.read_csv('data/symbipredict_2022.csv')
        medquad_df = pd.read_csv('data/medquad.csv')
        logging.info("Models and data loaded successfully.")
        return clf, le, symptom_columns, symptom_df, medquad_df
    except FileNotFoundError as e:
        logging.error(f"Error loading models or data: {e}")
        raise

def initialize_sentence_transformer():
    """Initializes the Sentence Transformer model with CUDA support if available."""
    logging.info("Initializing Sentence Transformer model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    logging.info("Sentence Transformer model initialized successfully.")
    return model

def initialize_chromadb(medquad_df):
    """Initializes ChromaDB and adds documents if the collection is empty."""
    logging.info("Initializing ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        if collection.count() == 0:
            logging.info("ChromaDB collection is empty. Adding documents...")
            medquad_df.dropna(subset=['question', 'answer', 'focus_area'], inplace=True)
            documents = [f"Question: {row['question']}\nFocus Area: {row['focus_area']}\n\n{row['answer']}" for index, row in medquad_df.iterrows()]
            ids = [str(i) for i in range(len(documents))]
            for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Adding documents to ChromaDB"):
                collection.add(
                    documents=documents[i:i+BATCH_SIZE],
                    ids=ids[i:i+BATCH_SIZE]
                )
            logging.info("Documents added to ChromaDB successfully.")
        else:
            logging.info("ChromaDB collection already exists and is not empty.")
        return collection
    except Exception as e:
        logging.error(f"Error initializing ChromaDB: {e}")
        raise

def get_top_symptoms(query, symptoms, model, threshold=SIMILARITY_THRESHOLD):
    """Finds the most similar symptoms to the user's query."""
    logging.info(f"Finding top symptoms for query: '{query}'")
    query_embedding = model.encode([query])
    symptom_embeddings = model.encode(symptoms)
    sim_scores = cosine_similarity(query_embedding, symptom_embeddings)[0]
    top_symptoms = []
    for i, score in enumerate(sim_scores):
        if score >= threshold:
            top_symptoms.append(symptoms[i])
    return top_symptoms

def predict_disease(symptoms, clf, le, symptom_columns):
    """Predicts the top 3 diseases based on the detected symptoms."""
    logging.info(f"Predicting diseases for symptoms: {symptoms}")
    input_vec = np.zeros(len(symptom_columns), dtype=int)
    for s in symptoms:
        if s in symptom_columns:
            idx = symptom_columns.index(s)
            input_vec[idx] = 1
    input_df = pd.DataFrame([input_vec], columns=symptom_columns)
    probas = clf.predict_proba(input_df)[0]
    top3_idx = probas.argsort()[-3:][::-1]
    top3_diseases = le.inverse_transform(top3_idx)
    top3_probs = probas[top3_idx]
    logging.info(f"Top 3 predicted diseases: {top3_diseases}")
    return top3_diseases, top3_probs

def retrieve_evidence(query, disease, collection, k=3):
    """Retrieves evidence from ChromaDB."""
    logging.info(f"Retrieving evidence for query: '{query}' and disease: '{disease}'")
    results = collection.query(
        query_texts=[query + " " + disease],
        n_results=k
    )
    logging.info(f"Retrieved {len(results['documents'][0])} documents from ChromaDB.")
    return results['documents']

def generate_explanation(query, symptoms, diseases, evidence):
    """Generates a user-friendly explanation using the Gemini API."""
    logging.info("Generating explanation with Gemini API...")
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logging.error("GEMINI_API_KEY environment variable not set.")
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = f"""
            You are a medical guidance assistant. Your task is to explain the predicted disease based on user symptoms and retrieved evidence.

            User Query: {query}
            Extracted Symptoms: {symptoms}
            Predicted Disease(s): {', '.join(diseases)}

            Retrieved Evidence Summary: {evidence}

            Instructions:
            1. Provide a concise and user-friendly explanation of the prediction.
            2. For each predicted disease:
            - Explain its likelihood based on the given symptoms.
            - Give a short, clear description of the disease (avoid technical jargon).
            3. Summarize how the evidence supports the prediction.
            4. Do **not** quote or copy text directly from the evidence; instead, paraphrase key insights.
            5. If the confidence seems low or multiple diseases are possible, mention that the user should consult a medical professional.

            Your output should be factual, empathetic, and easy for a non-expert user to understand.
        """


        response = model.generate_content(prompt)
        logging.info("Explanation generated successfully.")
        return response.text
    except Exception as e:
        logging.error(f"Error generating explanation with Gemini API: {e}")
        raise
