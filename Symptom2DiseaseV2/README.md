# Symptom to Disease Prediction with RAG

This project is an advanced symptom-to-disease prediction tool that leverages a machine learning model and a Retrieval-Augmented Generation (RAG) pipeline to provide accurate disease predictions and user-friendly explanations.

## Features

*   **Natural Language Input:** Describe your symptoms in plain English.
*   **Symptom Matching:** Uses a powerful Sentence Transformer model (`paraphrase-MiniLM-L6-v2`) to find the most relevant symptoms from a pre-defined list.
*   **Disease Prediction:** Employs a Random Forest classifier to predict the top 3 most likely diseases based on the detected symptoms.
*   **Retrieval-Augmented Generation (RAG):** Retrieves relevant medical information from a vector database (ChromaDB) to provide context and evidence for the predictions.
*   **Generative Explanations:** Uses the Gemini 1.5 Flash large language model to generate clear, concise, and user-friendly explanations of the results.
*   **Persistent Vector Database:** Uses a persistent ChromaDB instance to store the medical knowledge base, so it only needs to be built once.
*   **CUDA Support:** Automatically uses a CUDA-enabled GPU for faster inference if one is available.
*   **Comprehensive Logging:** Logs detailed information to both the console and a file for easy debugging and monitoring.

## Pipeline

The application follows a sophisticated pipeline to go from user input to final explanation:

1.  **Initialization:**
    *   Loads a pre-trained Random Forest model, a label encoder, and a list of known symptoms.
    *   Initializes the `paraphrase-MiniLM-L6-v2` Sentence Transformer model, using a CUDA device if available.
    *   Initializes a persistent ChromaDB vector database. On the first run, it populates the database with a medical knowledge base derived from the `medquad.csv` dataset. Each document in the database is a combination of a medical question, its focus area, and its answer, creating a rich context for retrieval.

2.  **Symptom Extraction:**
    *   When a user provides their symptoms in natural language, the application converts the query into a vector embedding.
    *   It then uses cosine similarity to compare the user's query embedding to the embeddings of all known symptoms.
    *   Symptoms with a similarity score above a certain threshold (currently 0.45) are selected.

3.  **Disease Prediction:**
    *   The list of selected symptoms is converted into a binary feature vector.
    *   This vector is fed into the Random Forest classifier, which outputs the probability for each possible disease.
    *   The top 3 diseases with the highest probabilities are selected.

4.  **Evidence Retrieval:**
    *   The application creates a search query by combining the user's original input and the top predicted disease.
    *   This combined query is used to search the ChromaDB database to find the most relevant medical documents.

5.  **Explanation Generation:**
    *   The user's query, the extracted symptoms, the predicted diseases, and the retrieved evidence are all passed to the Gemini 2.5 Flash model.
    *   The model generates a comprehensive, user-friendly explanation of the predictions, summarizing the evidence and providing context.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Symptom2DiseaseV2
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file:**
    *   In the root of the `Symptom2DiseaseV2` directory, create a file named `.env`.
    *   Add your Gemini API key to this file:
        ```
        GEMINI_API_KEY=YOUR_GEMINI_API_KEY
        ```

## Usage

To run the application, execute the following command from the `Symptom2DiseaseV2` directory:

```bash
python src/inference.py
```

The application will then prompt you to describe your symptoms.

### First Run

The first time you run the application, it will build the ChromaDB vector database. This may take a few minutes, and you will see a progress bar. This is a one-time process. Subsequent runs will be much faster as they will use the persistent database.

## Configuration

*   **`.env`:** Stores the `GEMINI_API_KEY`.
*   **`src/inference.py`:** Contains several constants that can be modified:
    *   `DB_PATH`: The path to the persistent ChromaDB database.
    *   `COLLECTION_NAME`: The name of the ChromaDB collection. Changing this will force the database to be rebuilt.
    *   `MODEL_NAME`: The name of the Sentence Transformer model to use.
    *   `SIMILARITY_THRESHOLD`: The cosine similarity threshold for symptom matching.
    *   `BATCH_SIZE`: The batch size for adding documents to ChromaDB.

## Logging

The application logs information to both the console and a file named `logs/app.log`. The log file contains more detailed information that can be useful for debugging.

## Future Improvements

*   **Hybrid Search:** Implement a hybrid search system that combines the current semantic search with a traditional keyword search (e.g., using BM25) to further improve retrieval accuracy.
*   **Web Interface:** Create a simple web interface (e.g., using Flask or FastAPI) to make the application more user-friendly.
*   **Model Retraining:** Implement a pipeline for retraining the Random Forest model with new data.
*   **More Comprehensive Evaluation:** Develop a more comprehensive evaluation framework to systematically test the accuracy of the symptom extraction and disease prediction components.
