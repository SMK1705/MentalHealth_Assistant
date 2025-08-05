from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from unified_guidance import generate_counselor_guidance
from logging_config import setup_logging
import logging

# Initialize logging early
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mental Health Counselor Guidance API",
    description="An API to generate mental health counseling guidance based on user input and conversation history.",
    version="1.0"
)

# Pydantic models for request and response
class GuidanceRequest(BaseModel):
    user_input: str
    patient_profile: Optional[Dict[str, Any]] = None
    conversation_history: Optional[str] = ""

class GuidanceResponse(BaseModel):
    generated_advice: str
    predicted_topic: str
    topic_confidence: float
    sentiment: str
    sentiment_score: float
    historical_examples: List[Dict[str, Any]]
    patient_profile: Dict[str, Any]


@app.get("/")
def root():
    return {"message": "Welcome to the Mental Health Counselor Guidance API!"}


@app.post("/guidance", response_model=GuidanceResponse)
def get_guidance(request: GuidanceRequest):
    try:
        guidance = generate_counselor_guidance(
            request.user_input,
            request.patient_profile,
            request.conversation_history,
        )
        return GuidanceResponse(
            generated_advice=guidance.get("generated_advice", ""),
            predicted_topic=guidance.get("predicted_topic", ""),
            topic_confidence=guidance.get("topic_confidence", 0.0),
            sentiment=guidance.get("sentiment", ""),
            sentiment_score=guidance.get("sentiment_score", 0.0),
            historical_examples=guidance.get("historical_examples", []),
            patient_profile=guidance.get("patient_profile", {}),
        )
    except Exception as e:
        logger.exception("Error generating guidance")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "OK"}
