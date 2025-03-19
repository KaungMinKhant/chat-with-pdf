from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import re
from datetime import datetime

# LangChain imports
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.schema import BaseRetriever
from pydantic import Field
from langchain.docstore.document import Document

# Import utility functions
from app.utils import (
    extract_text_from_pdf, split_text, create_documents, 
    create_vector_store, load_vector_store, save_vector_store,
    update_all_vector_store
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Chat With PDF API", 
              description="API for chatting with PDF documents using LLMs", 
              version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatQuery(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []

class ClearMemoryResponse(BaseModel):
    message: str
    success: bool

# Create a global conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Helper functions to reduce redundancy
def create_qa_chain(llm, retriever):
    """Create a conversational retrieval chain"""
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

def extract_sources_from_result(result) -> List[str]:
    """Extract unique source filenames from result"""
    sources = []
    source_documents = result.get("source_documents", [])
    for doc in source_documents:
        if doc.metadata and "source" in doc.metadata:
            source_path = doc.metadata["source"]
            source_name = Path(source_path).name
            if source_name not in sources:
                sources.append(source_name)
    return sources

def validate_document_ids(document_ids: List[str]) -> Tuple[List[str], List[str]]:
    """Validate documents and their indices exist, return lists of missing docs and indices"""
    missing_docs = []
    missing_indices = []
    
    for doc_id in document_ids:
        pdf_path = f"data/documents/{doc_id}.pdf"
        if not os.path.exists(pdf_path):
            missing_docs.append(doc_id)
            continue
            
        vectorstore_dir = f"data/vectorstores/{doc_id}"
        if not os.path.exists(vectorstore_dir):
            missing_indices.append(doc_id)
    
    return missing_docs, missing_indices

class StaticRetriever(BaseRetriever):
    """A retriever that returns a fixed set of documents"""
    stored_documents: List[Document] = Field(default_factory=list)
    
    def __init__(self, docs: List[Document], **kwargs):
        super().__init__(stored_documents=docs, **kwargs)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.stored_documents
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.stored_documents

@app.get("/")
def read_root():
    return {"message": "Welcome to the Chat With PDF API"}

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file for processing and indexing"""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Create data directories
        os.makedirs("data/documents", exist_ok=True)
        os.makedirs("data/vectorstores", exist_ok=True)
        
        # Generate a safe filename from the original filename
        original_filename = file.filename
        filename_only = os.path.basename(original_filename)
        base_name = os.path.splitext(filename_only)[0]
        safe_name = re.sub(r'[^\w\s-]', '_', base_name).strip()
        safe_name = re.sub(r'[-\s]+', '_', safe_name)
        
        document_id = safe_name
        pdf_path = f"data/documents/{document_id}.pdf"
        
        # Handle duplicate filenames
        if os.path.exists(pdf_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            document_id = f"{document_id}_{timestamp}"
            pdf_path = f"data/documents/{document_id}.pdf"
        
        # Save the PDF file
        with open(pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        try:
            # Process the PDF
            raw_text = extract_text_from_pdf(pdf_path)
            
            if not raw_text.strip():
                raise HTTPException(
                    status_code=422, 
                    detail="Could not extract text from the uploaded PDF"
                )
            
            text_chunks = split_text(raw_text)
            documents = create_documents(text_chunks, pdf_path)
            
            # Create individual vector store
            vectorstore = create_vector_store(documents)
            vectorstore_dir = f"data/vectorstores/{document_id}"
            save_vector_store(vectorstore, vectorstore_dir)
            
            # Update the combined "all" index
            all_index_path = "data/vectorstores/all"
            update_all_vector_store(documents, all_index_path)
            
            return {
                "message": "PDF uploaded successfully and indexed for searching",
                "document_id": document_id,
                "original_filename": original_filename,
                "chunks_processed": len(text_chunks)
            }
            
        except Exception as e:
            # Delete the saved PDF if processing fails
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/", response_model=ChatResponse)
async def chat_with_pdf(query: ChatQuery):
    """Send a query to chat with the uploaded PDF"""
    try:
        # Set up language model
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        llm = ChatOpenAI(temperature=0.0, model_name=model_name)
        
        # Check if specific document IDs are provided
        if query.document_ids and len(query.document_ids) > 0:
            # Validate documents exist
            missing_docs, missing_indices = validate_document_ids(query.document_ids)
            
            if missing_docs:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Documents not found: {', '.join(missing_docs)}"
                )
            
            if missing_indices:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Indices not found for documents: {', '.join(missing_indices)}"
                )
            
            # Load and combine selected vector stores
            all_docs = []
            
            for doc_id in query.document_ids:
                try:
                    vector_store = load_vector_store(f"data/vectorstores/{doc_id}")
                    docs = vector_store.similarity_search(query.query, k=3)
                    all_docs.extend(docs)
                except Exception as e:
                    logger.warning(f"Error loading index {doc_id}: {str(e)}")
            
            if not all_docs:
                return ChatResponse(
                    response="No relevant information found in the selected documents.",
                    sources=[]
                )
            
            # Create retriever and QA chain
            combined_retriever = StaticRetriever(all_docs)
            qa_chain = create_qa_chain(llm, combined_retriever)
            
        else:
            # Use the combined "all" vector store
            all_index_path = "data/vectorstores/all"
            
            if not os.path.exists(all_index_path):
                raise HTTPException(
                    status_code=404,
                    detail="No combined index found. Please index some PDFs first."
                )
            
            # Load combined index and create retriever
            all_vectorstore = load_vector_store(all_index_path)
            all_retriever = all_vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Create QA chain
            qa_chain = create_qa_chain(llm, all_retriever)
        
        # Process the query (common path for both cases)
        result = qa_chain.invoke({"question": query.query})
        sources = extract_sources_from_result(result)
        
        return ChatResponse(
            response=result["answer"],
            sources=sources
        )
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-memory/", response_model=ClearMemoryResponse)
async def clear_memory():
    """Clear the conversation memory"""
    try:
        global memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        return ClearMemoryResponse(
            message="Conversation memory cleared successfully",
            success=True
        )
    except Exception as e:
        return ClearMemoryResponse(
            message=f"Error clearing memory: {str(e)}",
            success=False
        )

@app.get("/memory/", response_model=Dict[str, Any])
async def get_memory():
    """Get the current conversation memory"""
    try:
        history = memory.chat_memory.messages
        history_dicts = messages_to_dict(history)
        
        return {"history": history_dicts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/")
async def list_documents():
    """List all uploaded documents"""
    try:
        documents = []
        if os.path.exists("data/documents"):
            for file in os.listdir("data/documents"):
                if file.endswith(".pdf"):
                    document_id = file.replace(".pdf", "")
                    is_indexed = os.path.exists(f"data/vectorstores/{document_id}")
                    
                    documents.append({
                        "document_id": document_id, 
                        "filename": file,
                        "is_indexed": is_indexed
                    })
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
