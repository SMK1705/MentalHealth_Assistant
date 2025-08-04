import logging
from langchain.schema import HumanMessage
from semantic_search import semantic_search
from model_cache import get_chat_groq
from prompt_templates import ADVICE_TEMPLATE

logger = logging.getLogger(__name__)

def generate_advice(query: str):
    logger.debug("Generating advice for query: %s", query)
    examples = semantic_search(query, top_k=3)
    
    examples_text = ""
    for ex in examples:
        examples_text += f"Patient: {ex.get('questionText', '')}\nTherapist: {ex.get('answerText', '')}\n\n"

    prompt = ADVICE_TEMPLATE.format(examples_text=examples_text, query=query)
    llm = get_chat_groq()
    response = llm.invoke([HumanMessage(content=prompt)])
    logger.debug("Advice generated: %s", response)
    return response.content if hasattr(response, "content") else str(response)
