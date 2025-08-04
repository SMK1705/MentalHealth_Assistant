from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from config import settings

@lru_cache(maxsize=1)
def get_embedding_model():
    """Return a cached embedding model instance."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@lru_cache(maxsize=1)
def get_chat_groq():
    """Return a cached ChatGroq instance."""
    return ChatGroq(
        temperature=0.7,
        model_name="llama-3.3-70b-versatile",
        groq_api_key=settings.groq_api_key,
    )
