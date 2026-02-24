from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def get_retriever():
    """Wrapper of Loads the vector DB retriever object for consumption"""
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("vectorstore/", embeddings)
    return db.as_retriever(search_type="similarity", k=4)