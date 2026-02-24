from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader,\
      CSVLoader, UnstructuredExcelLoader
import pandas as pd

def load_document(file_path):
    """
    function to load documents with langchain document loaders in multiple formats (PDF, TXT, CSV, Excel)
    """
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    elif file_path.endswith(".xls") or file_path.endswith(".xlsx"):
        loader = UnstructuredExcelLoader(file_path)
    else:
        raise ValueError("Unsupported file format")

    documents = loader.load()
    return documents


def chunk_documents(documents):
    """ function to split the loaded langchain document into smaller chuncks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)