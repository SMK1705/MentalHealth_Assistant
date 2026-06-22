import logging
from typing import Optional
from db import get_db
from schemas import Conversation, SessionLog

logger = logging.getLogger(__name__)

def _patient_profile_exists(db, patient_id: str) -> bool:
    """Check if a patient profile exists in the database."""

    return db["patients"].count_documents(
        {"patient_id": patient_id}, limit=1
    ) > 0


def archive_conversation(conversation: Conversation):
    conv_dict = conversation.dict()
    db = get_db()
    if not _patient_profile_exists(db, conversation.patient_id):
        raise ValueError(f"Patient profile {conversation.patient_id} does not exist")
    conversations_collection = db['PatientConvo']
    conversations_collection.replace_one(
        {"session_id": conversation.session_id},
        conv_dict,
        upsert=True
    )
    logger.info("Conversation %s archived successfully.", conversation.session_id)


def archive_session(log: SessionLog):
    """Archive analytical results for a counseling session."""

    log_dict = log.dict()
    db = get_db()
    if not _patient_profile_exists(db, log.patient_id):
        raise ValueError(f"Patient profile {log.patient_id} does not exist")
    sessions_collection = db["sessions"]
    sessions_collection.replace_one(
        {"session_id": log.session_id},
        log_dict,
        upsert=True,
    )
    logger.info("Session %s archived successfully.", log.session_id)


def load_session(session_id: str) -> Optional[SessionLog]:
    """Load a session log from the database with patient validation."""

    db = get_db()
    sessions_collection = db["sessions"]
    doc = sessions_collection.find_one({"session_id": session_id})
    if not doc:
        return None
    if not _patient_profile_exists(db, doc.get("patient_id", "")):
        raise ValueError(
            f"Patient profile {doc.get('patient_id')} does not exist"
        )
    return SessionLog(**doc)
