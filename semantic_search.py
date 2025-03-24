import pymongo
import logging
from config import settings
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

logger = logging.getLogger(__name__)

def semantic_search(query: str, top_k: int = 5):
    logger.debug("Starting semantic search for query: %s", query)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = embedding_model.embed_query(query)

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)
    response = index.query(
         namespace="default",
         vector=query_embedding,
         top_k=top_k,
         include_metadata=True
    )
    
    logger.debug("Pinecone response: %s", response)
    
    question_ids = []
    for match in response["matches"]:
         metadata = match.get("metadata", {})
         qid = metadata.get("questionID", None)
         if not qid:
             qid = match.get("id")
         question_ids.append(qid)
    logger.debug("Extracted question IDs: %s", question_ids)

    client = pymongo.MongoClient(settings.safe_mongo_uri)
    db = client.get_database("MentalHealthDB")
    collection = db["PatientConvo"]
    results = []
    for qid in question_ids:
         doc = collection.find_one({"questionID": str(qid)})
         if doc:
              results.append(doc)
    logger.debug("Semantic search found %d documents", len(results))
    return results
