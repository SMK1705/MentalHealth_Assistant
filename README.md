# Mental Health Counselor Guidance System

A comprehensive, AI-powered web application designed to assist mental health counselors by providing tailored guidance based on patient conversation data. The system combines data ingestion, natural language processing, machine learning, and LLM-based advice generation, all wrapped in an interactive chatbot-style user interface built with Streamlit.

## Overview

The goal of this project is to create a web-based tool that mental health counselors can use for on-demand guidance on how best to support patients. The application leverages historical counseling transcripts, semantic search, topic classification, sentiment analysis, and LLM (Retrieval-Augmented Generation) to surface actionable advice and generate detailed patient reports.

## Key Features

- **Data Ingestion & Storage:**  
  - Ingests mental health counseling transcripts from a MongoDB database.
  - Archives patient conversations to ensure persistent storage.

- **Natural Language Processing & Machine Learning:**  
  - **Semantic Search:** Uses HuggingFace embedding models and Pinecone for similarity search across counseling transcripts.
  - **Topic Classification:** Employs a zero-shot classifier to predict the primary topic of the patient’s conversation.
  - **Sentiment Analysis:** Implements a simple sentiment analysis to determine the overall emotional tone of the conversation.
  - **ML Model:** Predicts quantitative measures (e.g., upvotes/popularity) from counseling texts.

- **LLM-Based Guidance Generation:**  
  - Integrates retrieval-augmented generation (RAG) to provide context-aware, LLM-based advice.
  - Combines current user input with conversation history to generate tailored recommendations.

- **Interactive Chatbot UI:**  
  - A chat interface built with Streamlit that maintains conversation memory in session state.
  - Provides a single input field for a natural, continuous chat experience.
  - Includes an exit option that generates a comprehensive patient report, summarizing predicted topics, sentiment, and actionable recommendations.

## Project Structure

MentalHealthChatbot/
├── .env                    # Environment variables (not committed)
├── .streamlit/
│   └── config.toml         # Streamlit configuration (e.g., runOnSave = false)
├── Dockerfile              # (Optional) For container deployments
├── requirements.txt        # List of dependencies (Streamlit, pymongo, torch, etc.)
├── config.py               # Application configuration (loads .env)
├── logging_config.py       # Centralized logging configuration
├── schemas.py              # Pydantic models for messages and conversations
├── patient_profile.py      # Code to retrieve and update patient profiles
├── response_classifier.py  # Classifier for provider response types
├── safety.py               # Safety checker for red-flag terms and protocols
├── semantic_search.py      # Semantic search implementation using embeddings and Pinecone
├── topic_classifier.py     # Zero-shot topic classifier
├── unified_guidance.py     # Integrates conversation history, semantic search, and LLM advice generation
├── urgency_detector.py     # Detects urgency via emotion classification
├── vector_store.py         # Manages document embeddings and interactions with Pinecone
├── data_loader.py          # Loads and converts counseling transcripts from MongoDB
├── llm_rag.py              # LLM-based retrieval-augmented generation for advice
├── main.py                 # (Optional) Command-line interface for testing components
├── ml_model.py             # ML model for predicting upvotes/popularity
├── patient_ml.py           # Simple sentiment analysis for patient messages
├── archiver.py             # Archives conversations to the database
├── chatbot.py              # (Optional) Chatbot logic for interactive sessions
├── clustering.py           # Clusters patient problems using embeddings
└── app_chat.py             # Streamlit-based chatbot UI with conversation memory, exit button, and report generation


## Setup and Installation

### Prerequisites

- Python 3.10 or later
- MongoDB instance (local or hosted)
- Pinecone API key and index configuration (if using Pinecone for embeddings)
- API keys for any external LLM services (e.g., OpenAI, HuggingFace, or a local model)

### Installation Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/SMK1705/MentalHealth_Assistant.git
   cd mentalhealthchatbot

2. **Create and Activate a Virtual Environment:**

    python -m venv .venv
    source .venv/bin/activate   # On Windows use: .venv\Scripts\activate

3. **Install Dependencies:**

    pip install -r requirements.txt

4. **Configure Environment Variables:**

    Create a .env file in the project root with the necessary environment variables (e.g., MONGO_URI, GROQ_API_KEY, PINECONE_API_KEY, etc.).

5. **Optional – Configure Streamlit File Watcher:**

    Create a .streamlit/config.toml file with:
        [server]
        runOnSave = false

**Usage**
**Running the Chatbot Locally**

    python app_chat.py

*Start the Streamlit application:*

    streamlit run app_chat.py


This will open your default web browser at http://localhost:8501. You can then interact with the chatbot by typing messages into the input field. The conversation history is displayed dynamically, and clicking the "Exit Conversation and Show Report" button generates a detailed patient report that includes:

Predicted topic and confidence

Overall sentiment (and sentiment score)

Recommendations on steps the patient should take to overcome their challenges

**Testing the Backend Components**

You can also test individual modules (e.g., semantic search, ML model) by running their respective scripts (e.g., main.py, ml_model.py) or by using unit tests if available.