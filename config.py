from urllib.parse import quote_plus
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str
    mongo_uri: str  # original URI from .env
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_environment: str

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def safe_mongo_uri(self):
        import re
        match = re.match(r"(mongodb://)(.*):(.*)@(.*)", self.mongo_uri)
        if match:
            prefix, user, pwd, rest = match.groups()
            user_encoded = quote_plus(user)
            pwd_encoded = quote_plus(pwd)
            return f"{prefix}{user_encoded}:{pwd_encoded}@{rest}"
        return self.mongo_uri

settings = Settings()
