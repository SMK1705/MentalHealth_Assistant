import os

# Provide dummy settings so modules that build `config.Settings()` at import
# time can be imported during tests without a real .env file.
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("MONGO_URI", "mongodb://user:pass@localhost:27017")
os.environ.setdefault("PINECONE_API_KEY", "test-key")
os.environ.setdefault("PINECONE_INDEX_NAME", "test-index")
os.environ.setdefault("PINECONE_ENVIRONMENT", "us-east-1")
