import pymongo
import pandas as pd
import logging
from sklearn.cluster import KMeans
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings

logger = logging.getLogger(__name__)

def cluster_patient_problems(n_clusters=5):
    logger.info("Clustering patient problems into %d clusters", n_clusters)
    client = pymongo.MongoClient(settings.safe_mongo_uri)
    db = client.get_database("MentalHealthDB")
    collection = db["PatientConvo"]
    data = list(collection.find({}))
    logger.info("Loaded %d documents for clustering", len(data))
    df = pd.DataFrame(data)
    texts = df['questionText'].tolist()
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = [embedding_model.embed_query(text) for text in texts]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    df['cluster'] = clusters
    logger.info("Clustering completed.")
    return df[['questionText', 'topic', 'cluster']]
