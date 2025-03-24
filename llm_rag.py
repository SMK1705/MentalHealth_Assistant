import logging
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from config import settings
from semantic_search import semantic_search

logger = logging.getLogger(__name__)

def generate_advice(query: str):
    logger.debug("Generating advice for query: %s", query)
    examples = semantic_search(query, top_k=3)
    
    examples_text = ""
    for ex in examples:
         examples_text += f"Patient: {ex.get('questionText', '')}\nTherapist: {ex.get('answerText', '')}\n\n"
    
    prompt = f"""You are a mental health counselor. Based on the following examples, suggest advice:
Examples:
{examples_text}
New Query: {query}
Advice:"""
    llm = ChatGroq(
         temperature=0.7,
         model_name="llama-3.3-70b-versatile",
         groq_api_key=settings.groq_api_key
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    logger.debug("Advice generated: %s", response)
    return response
