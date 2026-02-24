from fastapi import FastAPI, UploadFile, File
from app.ingestion import load_document, chunk_documents
from app.embedding import create_vector_store
from app.retrieval import get_retriever
from app.rag_pipeline import create_rag_chain

app = FastAPI()

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    file_location = f"data/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    docs = load_document(file_location)
    chunks = chunk_documents(docs)
    create_vector_store(chunks)

    return {"message": "File processed successfully"}

@app.get("/query/")
def query(q: str):
    retriever = get_retriever()
    rag = create_rag_chain(retriever)
    response = rag.run(q)
    return {"response": response}