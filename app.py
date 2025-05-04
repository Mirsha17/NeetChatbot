# app.py

import os
import streamlit as st
from modules.vector_store import VectorStore
from modules.qa_pipeline import QAPipeline
from modules.chatbot_ui import ChatBotUI
from modules.pdf_handler import PDFHandler  # Make sure this exists


# Ensure st.set_page_config() is the first Streamlit function to run
st.set_page_config(page_title="NEET Subject Chatbot", page_icon="📚", layout="centered")


# Paths
#PDF_PATH = "data/biology.pdf"
#INDEX_PATH = "data/cbse_biology.index"
#METADATA_PATH = "data/cbse_biology_metadata.json"

# Paths for different subjects
PDF_PATHS = {
    "Physics": "data/Physics.pdf",
    "Chemistry": "data/Chemistry.pdf",
    "Biology": "data/biology.pdf"
}

INDEX_PATHS = {
    "Physics": "data/Physics.index",
    "Chemistry": "data/Chemistry.index",
    "Biology": "data/cbse_biology.index"
}

METADATA_PATHS = {
    "Physics": "data/Physics_metadata.json",
    "Chemistry": "data/Chemistry_metadata.json",
    "Biology": "data/cbse_biology_metadata.json"
}

# Initialize components
vector_store = VectorStore()
qa_pipeline = QAPipeline()

# Subject selection in Streamlit UI
selected_subject = st.selectbox("Select Subject", ["Physics", "Chemistry", "Biology"])
# Check if index exists for selected subject
if not os.path.exists(INDEX_PATHS[selected_subject]) or not os.path.exists(METADATA_PATHS[selected_subject]):
    st.info(f"📚 Index not found for {selected_subject}. Building index from PDF...")
    pdf_text = PDFHandler.extract_text(PDF_PATHS[selected_subject])
    text_chunks = pdf_text.split("\n\n")
    vector_store.build_index(text_chunks)
    vector_store.save_index(INDEX_PATHS[selected_subject], METADATA_PATHS[selected_subject])
else:
    vector_store.load_index(INDEX_PATHS[selected_subject], METADATA_PATHS[selected_subject])


# Start chatbot UI
chat_ui = ChatBotUI(qa_pipeline, vector_store,selected_subject)
chat_ui.render()
