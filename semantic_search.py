import logging
from model_cache import get_embedding_model, get_pinecone_index
from db import get_db

logger = logging.getLogger(__name__)


def _to_qid(value):
    """Normalize a questionID. Pinecone stores it as a float (189.0); MongoDB
    stores it as an int (189). Coerce both to int so they match."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return value


def semantic_search(query: str, top_k: int = 5):
    logger.debug("Starting semantic search for query: %s", query)
    query_embedding = get_embedding_model().embed_query(query)

    response = get_pinecone_index().query(
        namespace="default",
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )
    logger.debug("Pinecone response: %s", response)

    question_ids = []
    for match in response["matches"]:
        metadata = match.get("metadata") or {}
        qid = metadata.get("questionID")
        if qid is None:
            qid = match.get("id")
        qid = _to_qid(qid)
        if qid is not None:
            question_ids.append(qid)
    logger.debug("Extracted question IDs: %s", question_ids)

    # Pinecone gives float IDs, Mongo stores ints — query both forms, then
    # re-order results to match the Pinecone ranking (single batched lookup).
    variants = []
    for qid in question_ids:
        variants.append(qid)
        if isinstance(qid, int):
            variants.append(float(qid))
    collection = get_db()["PatientConvo"]
    docs = {_to_qid(d.get("questionID")): d for d in collection.find({"questionID": {"$in": variants}})}
    results = [docs[qid] for qid in question_ids if qid in docs]
    logger.debug("Semantic search found %d documents", len(results))
    return results
