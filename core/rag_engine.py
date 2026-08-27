import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_reteriver


def get_mistral_llm():

    return ChatMistralAI(
        model_name="mistral-small-2603",
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3
    )


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    reteriver = get_reteriver(vector_store=vector_store, k=4)
    llm = get_mistral_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """
            You're an expert meeting assistant. Answer the user's question based only on the meeting transcript contexrt provided below.

            If the answer is not found in the context, Say: 'I could nod find this information in the meeting transcript.'

            Always be consise and precise. If quoting someone, mention it clearly.

            Context from meeting transcript:
            {context}
        """),
        ("human", "{question}")
    ])

    # full LCEL Rag Pipeline
    rag_chain = (
        {
            "context": reteriver | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt | llm | StrOutputParser()
    )

    return rag_chain


def load_rag_chain():
    vector_store = load_vector_store()
    reteriver = get_reteriver()

    llm = get_mistral_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You're an expert meeting assistant. Answer the user's question based only on the meeting transcript contexrt provided below.
            
            If the answer is not found in the context, Say: 'I could nod find this information in the meeting transcript.'

            Always be consise and precise. If quoting someone, mention it clearly.

            Context from meeting transcript:
            {context}
        """),
        ()
    ])

    rag_chain = (
        {
            "context": reteriver | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt | llm | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    print(f"Question: ", {question})
    answer = rag_chain.invoke(question)
    print(f"Answer: ", {answer})
    return answer


