from langchain.chat_models import ChatOpenAI
from langchain_community.chains import RetrievalQA

def create_rag_chain(retriever):
    """
    Create a Retrieval-Augmented Generation (RAG) chain using a retriever and a GPT-4 language model.

    This function initializes a ChatOpenAI instance with the GPT-4 model and builds a RetrievalQA chain.
    The chain leverages the provided retriever to fetch relevant documents and uses the LLM to generate
    answers based on both the retrieved context and its own reasoning. Source documents are returned
    alongside the answers for transparency and traceability.

    Parameters
    ----------
    retriever : BaseRetriever
        A retriever object (e.g., vector store retriever) that supplies relevant documents
        for the query.

    Returns
    -------
    RetrievalQA
        A RetrievalQA chain configured with GPT-4 and the given retriever, capable of answering
        questions with context-aware responses and returning source documents.
    """

    llm = ChatOpenAI(model="gpt-4")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain