import pymongo
import pandas as pd
import logging
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from config import settings

logger = logging.getLogger(__name__)

def load_data_from_mongodb():
    client = pymongo.MongoClient(settings.safe_mongo_uri)
    db = client.get_database("MentalHealthDB")
    collection = db["PatientConvo"]
    data = list(collection.find({}))
    logger.info("Loaded %d documents from MongoDB", len(data))
    return data

def train_upvotes_model():
    data = load_data_from_mongodb()
    df = pd.DataFrame(data)
    df['upvotes'] = pd.to_numeric(df['upvotes'], errors='coerce')
    df = df.dropna(subset=['upvotes', 'questionText'])
    X = df['questionText']
    y = df['upvotes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline_model = make_pipeline(TfidfVectorizer(), LinearRegression())
    pipeline_model.fit(X_train, y_train)
    score = pipeline_model.score(X_test, y_test)
    logger.info("Model trained. R^2 score on test set: %.2f", score)
    return pipeline_model

def predict_upvotes(model, question_text):
    prediction = model.predict([question_text])
    logger.debug("Predicted upvotes: %s for question: %s", prediction[0], question_text)
    return prediction[0]
