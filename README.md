# Mental Health Counselor Guidance System

An AI-powered web application designed to assist mental health counselors by providing tailored guidance based on patient conversation data. This system integrates data ingestion, natural language processing, machine learning, and LLM-based advice generation, all within an interactive chatbot interface built using Streamlit.

## Overview

This project aims to empower mental health counselors with a web-based tool that delivers on-demand, actionable insights to better support patients. By leveraging historical counseling transcripts, semantic search, topic classification, sentiment analysis, and Retrieval-Augmented Generation (RAG), the application generates detailed patient reports and personalized recommendations.

## Key Features

- **Data Management:**  
    - Seamlessly ingests and archives mental health counseling transcripts from a MongoDB database.

- **Natural Language Processing & Machine Learning:**  
    - **Semantic Search:** Leverages HuggingFace embedding models and Pinecone for efficient similarity search.  
    - **Topic Classification:** Uses zero-shot classification to identify key conversation topics.  
    - **Sentiment Analysis:** Evaluates emotional tone using a sentiment analysis model.  
    - **ML Predictions:** Predicts quantitative measures (e.g., popularity/upvotes) from counseling texts.

- **LLM-Based Guidance:**  
    - Combines user input with conversation history using RAG to generate context-aware, tailored advice.

- **Interactive Chatbot Interface:**  
    - Built with Streamlit, offering conversation memory and a natural chat experience.  
    - Includes an "Exit Conversation" feature to generate a comprehensive patient report summarizing topics, sentiment, and recommendations.

## Project Structure

```bash
MentalHealthChatbot/
├── .env                    # Environment variables (not committed)
├── .streamlit/
│   └── config.toml         # Streamlit configuration
├── Dockerfile              # For container deployments (optional)
├── requirements.txt        # Dependencies (Streamlit, pymongo, torch, etc.)
├── config.py               # Application configuration
├── logging_config.py       # Centralized logging setup
├── schemas.py              # Pydantic models for data validation
├── patient_profile.py      # Manages patient profiles
├── response_classifier.py  # Classifies provider response types
├── safety.py               # Detects red-flag terms and ensures safety protocols
├── semantic_search.py      # Embedding-based semantic search
├── topic_classifier.py     # Zero-shot topic classification
├── unified_guidance.py     # Combines NLP and LLM for advice generation
├── urgency_detector.py     # Detects urgency in conversations
├── vector_store.py         # Manages embeddings and Pinecone interactions
├── data_loader.py          # Loads transcripts from MongoDB
├── llm_rag.py              # Implements RAG for advice generation
├── main.py                 # CLI for testing components
├── ml_model.py             # Predicts popularity/upvotes
├── patient_ml.py           # Sentiment analysis for patient messages
├── archiver.py             # Archives conversations to the database
├── chatbot.py              # Chatbot logic (optional)
├── clustering.py           # Clusters patient issues using embeddings
└── app_chat.py             # Streamlit chatbot UI with memory and reporting
```

## Setup and Installation

### Prerequisites

- Python 3.10 or later
- MongoDB instance (local or hosted)
- Pinecone API key and index configuration (if using Pinecone)
- API keys for external LLM services (e.g., OpenAI, HuggingFace)

### Installation Steps

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/SMK1705/MentalHealth_Assistant.git
    cd mentalhealthchatbot
    ```

2. **Create and Activate a Virtual Environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate   # On Windows: .venv\Scripts\activate
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Configure Environment Variables:**

    Create a `.env` file in the project root with required variables (e.g., `MONGO_URI`, `PINECONE_API_KEY`, etc.).

5. **Optional – Configure Streamlit File Watcher:**

    Add the following to `.streamlit/config.toml`:

    ```toml
    [server]
    runOnSave = false
    ```

## Usage

### Running the Chatbot Locally

Start the backend:

```bash
python main.py
```

Launch the Streamlit application:

```bash
streamlit run app_chat.py
```

This opens the chatbot interface in your browser at `http://localhost:8501`. Interact with the chatbot by typing messages. Click "Exit Conversation and Show Report" to generate a detailed patient report, including:

- Predicted topics and confidence levels  
- Sentiment analysis and scores  
- Actionable recommendations for patient challenges  

### Testing Backend Components

Test individual modules (e.g., semantic search, ML model) by running their respective scripts or using unit tests if available.

## Deployment

### Streamlit Community Cloud

1. Push your code to a GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Create a new app by selecting your repository and branch.
4. Set the main file to `app_chat.py`.
5. Configure environment variables if needed.
6. Deploy your app.

### Docker (Optional)

For containerized deployment, use the provided Dockerfile to build a Docker image and deploy on platforms that support Docker containers (e.g., Heroku, Render, Fly.io).

## Dependencies

- **Streamlit:** For building the interactive chatbot UI.
- **Pymongo:** For interfacing with MongoDB.
- **Torch:** Used with HuggingFace models for embeddings and NLP tasks.
- **LangChain and Transformers:** For LLM-based guidance and classification tasks.
- **Pydantic:** For data validation and modeling.
- Additional libraries: See `requirements.txt` for the complete list.

## Known Issues and Future Improvements

- **Performance:**  
    LLM calls and semantic search can be resource-intensive. Future iterations could incorporate asynchronous processing or caching to improve response times.

- **UI Enhancements:**  
    Further UI improvements (e.g., chat bubbles, emojis, real-time updates) and better error handling can enhance the user experience.

- **Scalability:**  
    Evaluate deploying on platforms that support scaling to handle increased user demand.

- **Security:**  
    Strengthen security by managing secrets via dedicated services and ensuring secure communication between components.
