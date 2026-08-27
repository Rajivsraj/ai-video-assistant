"""
Actionable Items, decisions, questions
"""

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os


def get_mistral_llm():
    return ChatMistralAI(
        model_name="mistral-small-2603",
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3
    )


def build_chain(system_prompt: str):
    llm = get_mistral_llm()
    return (
        RunnablePassthrough() |
        RunnableLambda(lambda x: {"text": x}) |
        ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}")
        ]) | llm | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "you're an expert meeting analyst."
        "From the meeting transcript, extract all the action items for each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not Specified')\n\n"
        "Format as a numbered list. If not found say 'No action items found'"
    )

    return chain.invoke(transcript)


def extract_key_decision(transcript: str) -> str:
    chain = build_chain(
        "you're an expert meeting analyst. From the meeting transcript,"
        "extract all key decisions made. Format as a numbered list."
        "If none found say 'No key decision found'"
    )

    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "you're an expert meeting analyst. From the meeting transcript,"
        "extract all unresolved questions or topics needs follow-up. Format as a numbered list."
        "If none found say 'No open questions found'"
    )

    return chain.invoke(transcript)


