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
    conversation_history: Optional[str] = ""

class GuidanceResponse(BaseModel):
    generated_advice: str
    predicted_topic: str
    topic_confidence: float
    historical_examples: List[Dict[str, Any]]


@app.get("/")
def root():
    return {"message": "Welcome to the Mental Health Counselor Guidance API!"}


@app.post("/guidance", response_model=GuidanceResponse)
def get_guidance(request: GuidanceRequest):
    try:
        guidance = generate_counselor_guidance(request.user_input, request.conversation_history)
        return GuidanceResponse(
            generated_advice=guidance.get("generated_advice", ""),
            predicted_topic=guidance.get("predicted_topic", ""),
            topic_confidence=guidance.get("topic_confidence", 0.0),
            historical_examples=guidance.get("historical_examples", [])
        )
    except Exception as e:
        logger.exception("Error generating guidance")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "OK"}
