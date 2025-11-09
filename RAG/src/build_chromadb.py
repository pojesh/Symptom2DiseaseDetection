# src/build_chromadb.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

# Load MedQuAD CSV
df = pd.read_csv('C:/Users/Pojesh/Documents/GitHub/Symptom2DiseaseDetection/RAG/data/medquad.csv')

# Prepare corpus for embedding
corpus_list = [
    f"Q: {row['question']}\nA: {row['answer']}\nArea: {row['focus_area']}"
    for _, row in df.iterrows()
]

# Initialize
client = chromadb.PersistentClient(path="data/chroma_db_store")
collection = client.create_collection(name="medquad_collection")

# ClinicalBERT sentence transformer
model = SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT')


for idx, doc in tqdm(enumerate(corpus_list), total=len(corpus_list)):
    embedding = model.encode(doc)
    collection.add(embeddings=[embedding], documents=[doc], ids=[str(idx)])


print("ChromaDB build complete: {} docs embedded.".format(len(corpus_list)))
