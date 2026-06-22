import logging
from langchain.schema import HumanMessage
from semantic_search import semantic_search
from model_cache import get_chat_groq
from prompt_templates import ADVICE_TEMPLATE

logger = logging.getLogger(__name__)


def _build_prompt(query: str, examples=None) -> str:
    if examples is None:
        examples = semantic_search(query, top_k=3)
    examples_text = ""
    for ex in examples:
        examples_text += f"Patient: {ex.get('questionText', '')}\nTherapist: {ex.get('answerText', '')}\n\n"
    return ADVICE_TEMPLATE.format(examples_text=examples_text, query=query)


def generate_advice(query: str, examples=None):
    """Return the full advice text (blocking). Used by the API/CLI callers."""
    logger.debug("Generating advice for query: %s", query)
    prompt = _build_prompt(query, examples)
    response = get_chat_groq().invoke([HumanMessage(content=prompt)])
    logger.debug("Advice generated: %s", response)
    return response.content if hasattr(response, "content") else str(response)


def stream_advice(query: str, examples=None):
    """Yield advice text chunks as the LLM produces them (for st.write_stream)."""
    logger.debug("Streaming advice for query: %s", query)
    prompt = _build_prompt(query, examples)
    for chunk in get_chat_groq().stream([HumanMessage(content=prompt)]):
        text = getattr(chunk, "content", None)
        if text:
            yield text
