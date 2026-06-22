from functools import lru_cache
import pymongo
from config import settings

DB_NAME = "MentalHealthDB"
# RAG knowledge corpus, kept separate from the PatientConvo conversation archive.
CORPUS_COLLECTION = "corpus"


@lru_cache(maxsize=1)
def get_mongo_client():
    """Return a process-wide cached MongoClient.

    pymongo manages an internal connection pool, so a single client should be
    created once and reused across calls. Creating a new client per operation
    (the previous pattern) spins up a fresh pool each time and leaks
    connections when the client is never closed.
    """
    return pymongo.MongoClient(settings.safe_mongo_uri)


def get_db(name: str = DB_NAME):
    """Return the application database from the shared client."""
    return get_mongo_client().get_database(name)
