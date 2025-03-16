from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid

app = FastAPI(title="Chat With PDF API", 
              description="API for chatting with PDF documents using LLMs", 
              version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatQuery(BaseModel):
    query: str
    document_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []

@app.get("/")
def read_root():
    return {"message": "Welcome to the Chat With PDF API"}

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file for processing"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Generate a unique ID for the document
    document_id = str(uuid.uuid4())
    
    # Create data directory if it doesn't exist
    os.makedirs("data/documents", exist_ok=True)
    
    # Save the PDF file
    file_path = f"data/documents/{document_id}.pdf"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # TODO: Add code to process and index the PDF
    
    return {"message": "PDF uploaded successfully", "document_id": document_id}

@app.post("/chat/", response_model=ChatResponse)
async def chat_with_pdf(query: ChatQuery):
    """Send a query to chat with the uploaded PDF"""
    if query.document_id and not os.path.exists(f"data/documents/{query.document_id}.pdf"):
        raise HTTPException(status_code=404, detail="Document not found")
    
    # TODO: Implement actual chat functionality with LangChain/LlamaIndex
    
    # Placeholder response
    return ChatResponse(
        response=f"This is a placeholder response to your query: '{query.query}'",
        sources=[]
    )

@app.get("/documents/")
async def list_documents():
    """List all uploaded documents"""
    try:
        documents = []
        if os.path.exists("data/documents"):
            for file in os.listdir("data/documents"):
                if file.endswith(".pdf"):
                    document_id = file.replace(".pdf", "")
                    documents.append({"document_id": document_id, "filename": file})
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
