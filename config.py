from urllib.parse import quote_plus, unquote
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str
    mongo_uri: str  # original URI from .env
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_environment: str
    app_password: str = ""  # optional shared password gating the Streamlit app

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def safe_mongo_uri(self):
        import re
        # Username has no unescaped ':'/'@', the password runs up to the LAST
        # '@', and the host is the final '@'-free segment. This avoids the
        # greedy mis-split a password containing ':' would otherwise cause.
        match = re.match(
            r"(mongodb(?:\+srv)?://)([^:@/]+):(.*)@([^@]+)$", self.mongo_uri
        )
        if match:
            prefix, user, pwd, rest = match.groups()
            # Decode first, then re-encode, so an already-encoded password is
            # not double-encoded (%40 -> %2540) and a raw one is encoded correctly.
            user_encoded = quote_plus(unquote(user))
            pwd_encoded = quote_plus(unquote(pwd))
            return f"{prefix}{user_encoded}:{pwd_encoded}@{rest}"
        return self.mongo_uri

settings = Settings()
