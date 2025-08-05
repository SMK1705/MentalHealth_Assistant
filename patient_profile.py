import time
import json
import pymongo
import logging
from config import settings
from pinecone import Pinecone, ServerlessSpec
from model_cache import get_embedding_model
from schemas import PatientProfile

logger = logging.getLogger(__name__)

def get_patient_conversation(patient_id: str):
    client = pymongo.MongoClient(settings.safe_mongo_uri)
    db = client.get_database("MentalHealthDB")
    collection = db['PatientConvo']
    conv = collection.find_one({"patient_id": patient_id})
    return conv


def get_patient_profile(patient_id: str) -> PatientProfile | None:
    client = pymongo.MongoClient(settings.safe_mongo_uri)
    db = client.get_database("MentalHealthDB")
    collection = db["patients"]
    data = collection.find_one({"patient_id": patient_id})
    if not data:
        return None

    medical_history = data.get("medical_history", [])
    therapy_goals = data.get("therapy_goals", [])

    if isinstance(medical_history, str):
        try:
            medical_history = json.loads(medical_history)
        except json.JSONDecodeError:
            medical_history = [medical_history]
    if isinstance(therapy_goals, str):
        try:
            therapy_goals = json.loads(therapy_goals)
        except json.JSONDecodeError:
            therapy_goals = [therapy_goals]

    data["medical_history"] = medical_history if isinstance(medical_history, list) else [medical_history]
    data["therapy_goals"] = therapy_goals if isinstance(therapy_goals, list) else [therapy_goals]

    collection.update_one(
        {"patient_id": patient_id},
        {"$set": {
            "medical_history": data["medical_history"],
            "therapy_goals": data["therapy_goals"]
        }}
    )

    return PatientProfile(**data)

def update_patient_profile(patient_id: str):
    conv = get_patient_conversation(patient_id)
    profile = get_patient_profile(patient_id)
    if not conv and not profile:
        logger.warning("No data found for patient %s", patient_id)
        return

    patient_texts = []
    if conv:
        for msg in conv.get("messages", []):
            if msg.get("is_user"):
                patient_texts.append(msg.get("content", ""))

    if profile:
        patient_texts.extend(profile.medical_history)
        patient_texts.extend(profile.therapy_goals)

    if not patient_texts:
        logger.warning("No patient messages found in conversation for patient %s", patient_id)
        return
    profile_text = "\n".join(patient_texts)

    embedding_model = get_embedding_model()
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
