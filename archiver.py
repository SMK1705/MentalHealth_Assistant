import pymongo
import logging
from config import settings
from schemas import Conversation

logger = logging.getLogger(__name__)
client = pymongo.MongoClient(settings.safe_mongo_uri)
db = client.get_database("MentalHealthDB")
conversations_collection = db['PatientConvo']

def archive_conversation(conversation: Conversation):
    conv_dict = conversation.dict()
    conversations_collection.replace_one(
        {"session_id": conversation.session_id},
        conv_dict,
        upsert=True
    )
    logger.info("Conversation %s archived successfully.", conversation.session_id)
