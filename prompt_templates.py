from langchain.prompts import ChatPromptTemplate

ADVICE_TEMPLATE = ChatPromptTemplate.from_template(
    "You are a mental health counselor. Based on the following examples, suggest advice.\n"
    "Examples:\n{examples_text}\n"
    "New Query: {query}\n\n"
    "Respond using the following structure:\n"
    "Advice: <clear recommendation>\n"
    "Rationale: <brief justification>\n"
    "Suggested Actions: <numbered steps>\n"
)
