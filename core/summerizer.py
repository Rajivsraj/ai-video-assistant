from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os
from dotenv import load_dotenv
"""
open-mistral-nemo → 12B model, 128k context
open-mistral-7b → 7B model, 32k context
open-mixtral-8x7b → Mixture‑of‑Experts (56B active params), 32k context
open-mixtral-8x22b → Larger MoE (141B active params), 64k context
codestral-latest → 22B, optimized for code generation"""

load_dotenv()

def get_mistral_llm():

    return ChatMistralAI(
        model_name="mistral-small-2603",
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_mistral_llm()

    map_prompt = ChatPromptTemplate(
        [
            ("system", "Summarize this portion of a meeting transcript preciously"),
            ("human", "{text}")
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()
    chunks = split_transcript(transcript=transcript)

    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]
    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate([
        ("system", "You're an expert meeting summarizer. Combine these partial summaries into one final professional meeting in bullet points"),
        ("human", "{text}")
    ])

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text":x}) | combined_prompt | llm | StrOutputParser
    )

    return combined_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    llm = get_mistral_llm()

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) |
        ChatPromptTemplate([
            ("system", "Based on meeting transcript generate a short professional meeting title. Return title only nothing else."),
            ("human", "{text}")
        ]) | llm | StrOutputParser()
    )

    return title_chain.invoke(transcript[:200])

