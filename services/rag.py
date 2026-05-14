from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from core.config import EMBED_MODEL, CHAT_MODEL, CHROMA_PATH, COLLECTION_NAME

def ask_question(question: str) -> str:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 4, "score_threshold": 0.4}
    )

    prompt = ChatPromptTemplate.from_template("""
You are a personal health companion developed by Dilan. You are friendly, helpful, and knowledgeable about medical topics.

For greetings or personal questions (like "how are you", "who are you", "what can you do"), respond warmly and naturally without using the context.

For non-medical or off-topic questions (like coding, politics, sports, cooking etc.), respond with:
"I'm a health companion and can only assist with medical and health-related questions. Feel free to ask me anything about symptoms, conditions, or health advice!"

For medical questions, answer strictly based on the uploaded document context provided below.

If a medical question is not covered in the context, respond with:
"I don't have enough information in the uploaded document to answer that."

Do NOT make up medical answers. Do NOT use medical knowledge outside the provided context.

Context:
{context}

Question: {question}

Answer:
""")

    llm = ChatOllama(model=CHAT_MODEL)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)