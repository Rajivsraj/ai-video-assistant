from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os


# Ensure downloads folder is inside main project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # utils/
MAIN_DIR = os.path.dirname(BASE_DIR)                   # ai-video-assistant/
CHROMA_DIR = os.path.join(MAIN_DIR, "vector_db")
os.makedirs(CHROMA_DIR, exist_ok=True)

COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def build_vector_store(transcript: str) -> Chroma:
    print("Building Vector store")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )

    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i}) for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    return vector_store


def load_vector_store() -> Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vector_store


def get_reteriver(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )