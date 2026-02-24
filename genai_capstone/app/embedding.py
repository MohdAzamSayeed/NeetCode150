from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def create_vector_store(chunks):
    """create a vector database based on provided chunks and save in FAISS VectorDB"""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore/")