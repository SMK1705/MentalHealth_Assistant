from pydantic import BaseModel, Field
from datetime import datetime
from typing import List

class Message(BaseModel):
    content: str
    is_user: bool
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict = Field(default_factory=dict)

class Conversation(BaseModel):
    session_id: str
    patient_id: str = ""  # Patient identifier
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_message(self, message: Message):
        self.messages.append(message)
        self.updated_at = datetime.now()


class PatientProfile(BaseModel):
    patient_id: str
    medical_history: List[str] = Field(default_factory=list)
    therapy_goals: List[str] = Field(default_factory=list)
