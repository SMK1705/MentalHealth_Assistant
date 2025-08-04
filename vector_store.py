import time
import hashlib
import logging
from typing import Any, List
from config import settings
from langchain.schema import Document, BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from model_cache import get_embedding_model

logger = logging.getLogger(__name__)

def initialize_vector_store(docs):
    logger.debug("Initializing vector store with %d docs", len(docs))
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    split_docs = text_splitter.split_documents(docs)
    
    embedding_model = get_embedding_model()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    indexes = pc.list_indexes().names()
    if settings.pinecone_index_name not in indexes:
        logger.info("Creating Pinecone index: %s", settings.pinecone_index_name)
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=384,
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
    
    candidate_docs = {}
    for doc in split_docs:
        qid = doc.metadata.get("questionID", "")
        if qid:
            vector_id = str(qid)
        else:
            vector_id = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
        candidate_docs[vector_id] = doc

    candidate_ids = list(candidate_docs.keys())
    existing_ids = set()
    fetch_batch_size = 100
    for i in range(0, len(candidate_ids), fetch_batch_size):
        batch_ids = candidate_ids[i:i+fetch_batch_size]
        fetch_response = index.fetch(ids=batch_ids, namespace="default")
        if fetch_response.vectors:
            existing_ids.update(fetch_response.vectors.keys())
    
    new_vectors = []
    # Collect documents that are not yet in the index so we can
    # embed them in a single batch. This significantly reduces the
    # overhead of repeatedly calling the embedding model for each
    # document and speeds up initialization for large corpora.
    missing_docs = [
        (vector_id, doc)
        for vector_id, doc in candidate_docs.items()
        if vector_id not in existing_ids
    ]
    if missing_docs:
        texts = [doc.page_content for _, doc in missing_docs]
        embeddings = embedding_model.embed_documents(texts)
        for (vector_id, doc), embedding in zip(missing_docs, embeddings):
            meta = dict(doc.metadata) if doc.metadata else {}
            meta["text"] = doc.page_content
            new_vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": meta
            })
    
    upsert_batch_size = 100
    if new_vectors:
        logger.info("Upserting %d new vectors", len(new_vectors))
        for i in range(0, len(new_vectors), upsert_batch_size):
            batch = new_vectors[i:i+upsert_batch_size]
            index.upsert(vectors=batch, namespace="default")
    
    class PineconeRetriever(BaseRetriever):
        index: Any
        embedding_model: Any
        namespace: str

        class Config:
            arbitrary_types_allowed = True

        def get_relevant_documents(self, query: str, top_k: int = 3) -> List[Document]:
            query_embedding = self.embedding_model.embed_query(query)
            response = self.index.query(
                namespace=self.namespace,
                vector=query_embedding,
                top_k=top_k,
                include_values=True,
                include_metadata=True
            )
            docs = []
            for match in response["matches"]:
                meta = match.get("metadata", {})
                text = meta.get("text", "")
                docs.append(Document(page_content=text, metadata=meta))
            return docs

        @property
        def search_kwargs(self):
            return {"k": 3}

        def add_texts(self, texts, metadatas=None, **kwargs):
            raise NotImplementedError("This retriever does not support adding texts.")

        def similarity_search(self, query: str, k: int = 3, **kwargs) -> List[Document]:
            return self.get_relevant_documents(query, top_k=k)

    logger.debug("Vector store initialization complete")
    return PineconeRetriever(index=index, embedding_model=embedding_model, namespace="default")
