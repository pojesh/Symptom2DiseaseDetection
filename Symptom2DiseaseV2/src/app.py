
import streamlit as st
import logging
from inference import (
    load_models_and_data,
    initialize_chromadb,
    initialize_sentence_transformer,
    run_inference,
    load_config
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Symptom to Disease Predictor",
    page_icon="🩺",
    layout="wide"
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)

# --- Load Configuration ---
try:
    config = load_config()
    logging.info("Configuration loaded successfully.")
except FileNotFoundError:
    st.error("FATAL: config.yaml not found. The application cannot start.")
    logging.error("FATAL: config.yaml not found.")
    st.stop()


# --- Model and Data Loading ---
@st.cache_resource
def load_all():
    """Loads and initializes all models and data using functions from inference.py."""
    logging.info("Loading all resources...")
    try:
        models = load_models_and_data()
        _, _, _, medquad_df = models
        collection = initialize_chromadb(medquad_df)
        sentence_model = initialize_sentence_transformer()
        logging.info("All resources loaded successfully.")
        return models, collection, sentence_model
    except Exception as e:
        logging.error(f"An error occurred during resource loading: {e}")
        raise

# --- Main Application UI ---
def main():
    """Main function to run the Streamlit application."""
    st.title("🩺 Symptom to Disease Predictor")
    st.write("Describe your symptoms in the text box below. The system will analyze them to provide potential insights.")

    try:
        models, collection, sentence_model = load_all()
    except Exception as e:
        st.error("Failed to load necessary models and data. The application cannot proceed.")
        st.error(f"Details: {e}")
        return

    # Display the current operation mode based on config
    mode = "Random Forest Prediction" if config.get('use_rf_prediction', True) else "Direct Knowledge Search (RAG)"
    st.info(f"**Current Mode:** {mode}")

    user_nl = st.text_area("Describe your symptoms:", height=150, placeholder="e.g., 'I have a high fever, a persistent cough, and I'm feeling very tired.'")

    if st.button("Analyze Symptoms"):
        if not user_nl.strip():
            st.warning("Please describe your symptoms.")
            return

        with st.spinner("Analyzing..."):
            try:
                result = run_inference(user_nl, models, collection, sentence_model)

                # --- Display results based on the status ---
                if result['status'] == 'success':
                    st.subheader("Top Predictions")
                    for disease, probability in result['predictions']:
                        st.metric(label=disease, value=f"{probability*100:.2f}% likelihood")
                    
                    st.subheader("Explanation")
                    st.markdown(result['explanation'])

                elif result['status'] == 'fallback':
                    st.warning("Could not identify specific symptoms for prediction. Providing a general analysis based on your query.")
                    st.subheader("General Analysis")
                    st.markdown(result['explanation'])

                elif result['status'] == 'rag_only':
                    st.subheader("Analysis from Medical Knowledge Base")
                    st.markdown(result['explanation'])

            except Exception as e:
                st.error("An unexpected error occurred during analysis.")
                logging.error(f"An error occurred during inference: {e}")

if __name__ == "__main__":
    main()

