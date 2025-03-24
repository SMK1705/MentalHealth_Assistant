import ssl
import pymongo
import logging
from langchain.schema import Document
from config import settings

logger = logging.getLogger(__name__)

def load_dataset():
    """Load dataset from MongoDB (MentalHealthDB, PatientConvo)
    and convert records to LangChain Document objects."""
    try:
        client = pymongo.MongoClient(settings.safe_mongo_uri, ssl=True, ssl_cert_reqs=ssl.CERT_NONE)
        db = client.get_database("MentalHealthDB")
        collection = db["PatientConvo"]
        
        data = list(collection.find({}))
        logger.info("Number of documents found: %d", len(data))
        
        docs = [
            Document(
                page_content=f"Patient: {row.get('questionText', '')}\nCounselor: {row.get('answerText', '')}",
                metadata={
                    "questionID": str(row.get("questionID", "")),
                    "questionTitle": row.get("questionTitle", ""),
                    "topic": row.get("topic", ""),
                    "therapistInfo": row.get("therapistInfo", ""),
                    "upvotes": row.get("upvotes", ""),
                    "views": row.get("views", "")
                }
            )
            for row in data
        ]
        return docs
        
    except Exception as e:
        logger.error("Error loading dataset from MongoDB: %s", str(e))
        exit(1)
