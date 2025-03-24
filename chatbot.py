import logging
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from schemas import Conversation, Message
from safety import SafetyChecker
from typing import Optional, Tuple
from config import settings
from langchain.schema import HumanMessage

logger = logging.getLogger(__name__)

class MentalHealthAssistant:
    def __init__(self, vector_store, patient_id: str):
        self.llm = ChatGroq(
            temperature=0.7,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=settings.groq_api_key
        )
        logger.debug("LLM for chatbot initialized.")
        self.vector_store = vector_store
        self.safety = SafetyChecker()
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store,
            memory=self._create_memory(),
            chain_type="stuff",
            verbose=True
        )
        logger.debug("QA chain created for chatbot.")
        self.conversations = {}
        self.patient_id = patient_id
    
    def _create_memory(self):
        return ConversationSummaryBufferMemory(
            llm=self.llm,
            memory_key="chat_history",
            max_token_limit=2000,
            return_messages=True
        )
    
    def process_message(self, session_id: str, user_input: str) -> Tuple[str, Optional[dict]]:
        logger.debug("Processing message for session %s: %s", session_id, user_input)
        safety_result = self.safety.check_input(user_input)
        if safety_result:
            logger.warning("Safety check triggered for session %s: %s", session_id, safety_result)
            return safety_result["response"], {"safety_flag": safety_result}
        
        conversation = self._get_conversation(session_id)
        
        result = self.qa_chain({
            "question": user_input,
            "chat_history": [(msg.content, msg.content) for msg in conversation.messages[-4:]]
        })
        
        self._save_message(session_id, user_input, is_user=True)
        self._save_message(session_id, result["answer"], is_user=False)
        logger.debug("Message processed for session %s", session_id)
        return result["answer"], None
    
    def _get_conversation(self, session_id: str) -> Conversation:
        if session_id in self.conversations:
            return self.conversations[session_id]
        else:
            conv = Conversation(session_id=session_id, patient_id=self.patient_id)
            self.conversations[session_id] = conv
            logger.debug("New conversation created for session %s", session_id)
            return conv
    
    def _save_message(self, session_id: str, content: str, is_user: bool):
        conversation = self._get_conversation(session_id)
        message = Message(content=content, is_user=is_user)
        conversation.add_message(message)
        logger.debug("Saved message for session %s: %s", session_id, content)
    
    def summarize_conversation(self, session_id: str) -> str:
        conversation = self._get_conversation(session_id)
        conversation_text = "\n".join(
            f"{'User' if msg.is_user else 'Assistant'}: {msg.content}" 
            for msg in conversation.messages
        )
        prompt = f"Summarize the following conversation in a few sentences:\n{conversation_text}\nSummary:"
        summary = self.llm.invoke([HumanMessage(content=prompt)])
        logger.debug("Conversation summary for session %s: %s", session_id, summary)
        return summary
