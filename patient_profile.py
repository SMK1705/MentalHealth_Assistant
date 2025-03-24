import time
import pymongo
import logging
from config import settings
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)

def get_patient_conversation(patient_id: str):
    client = pymongo.MongoClient(settings.safe_mongo_uri)
    db = client.get_database("MentalHealthDB")
    collection = db['PatientConvo']
    conv = collection.find_one({"patient_id": patient_id})
    return conv

def update_patient_profile(patient_id: str):
    conv = get_patient_conversation(patient_id)
    if not conv:
        logger.warning("No conversation found for patient %s", patient_id)
        return

    patient_texts = []
    for msg in conv.get("messages", []):
        if msg.get("is_user"):
            patient_texts.append(msg.get("content", ""))
    if not patient_texts:
        logger.warning("No patient messages found in conversation for patient %s", patient_id)
        return
    profile_text = "\n".join(patient_texts)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embedding = embedding_model.embed_query(profile_text)

    pc = Pinecone(api_key=settings.pinecone_api_key)
    indexes = pc.list_indexes().names()
    if settings.pinecone_index_name not in indexes:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=len(embedding),
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region=settings.pinecone_environment
            )
        )
        while True:
            index_status = pc.describe_index(settings.pinecone_index_name)
            if index_status.status['ready']:
                break
            time.sleep(1)
    index = pc.Index(settings.pinecone_index_name)
    vector = {
        "id": patient_id,
        "values": embedding,
        "metadata": {"patient_id": patient_id, "profile_text": profile_text}
    }
    index.upsert(vectors=[vector], namespace="patients")
    logger.info("Patient profile for %s updated in Pinecone.", patient_id)
