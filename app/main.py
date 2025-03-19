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
from langchain_openai import ChatOpenAI
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.schema import BaseRetriever
from pydantic import Field
from langchain.docstore.document import Document
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import BaseCallbackHandler

# Import utility functions
from app.utils import (
    extract_text_from_pdf, split_text, create_documents, 
    create_vector_store, load_vector_store, save_vector_store,
    update_all_vector_store
)
import hashlib

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

# Create global memory object
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=5,
    input_key="input",
    output_key="output"
)

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

def create_retrieval_tool(retriever, name="document_search", description=None):
    """Create a tool for document retrieval"""
    # Sanitize name to meet OpenAI's function name requirements
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    
    # Ensure name is within OpenAI's 64-character limit
    if len(sanitized_name) > 60:
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        sanitized_name = f"doc_{name_hash}"
    
    if description is None:
        description = (
            "Searches the documents to find answers to questions about their content. "
            "Use this tool whenever you need to access specific information from the documents. "
            "ALWAYS USE THIS TOOL FIRST before answering any question."
        )
    
    def retrieval_with_sources(query):
        """Retrieve documents and include sources in the response"""
        docs = retriever.invoke(query)
        results = []
        sources = set()
        
        for doc in docs:
            source_path = doc.metadata.get("source", "unknown")
            source_name = Path(source_path).name
            sources.add(source_name)
            results.append(f"[Content from {source_name}]: {doc.page_content}")
        
        # Create sources section
        sources_list = list(sources)
        source_list_formatted = "\n".join([f"- {s}" for s in sources_list])
        source_section = f"""

            !!! DOCUMENT SOURCES !!!
            {source_list_formatted}

            ⚠️ YOU MUST END YOUR RESPONSE WITH:

            SOURCES:
            {source_list_formatted}

            FAILURE TO INCLUDE THE SOURCES SECTION EXACTLY AS SHOWN ABOVE WILL RESULT IN REJECTION.
        """
        results.append(source_section)
        
        # Store the sources for later use
        retrieved_sources = {"sources": sources_list}
        
        return "\n\n".join(results), retrieved_sources
    
    return Tool(
        name=sanitized_name,
        description=description,
        func=retrieval_with_sources,
        return_direct=False
    )

def create_agent(llm, tools):
    """Create an agent with a set of tools"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant for question-answering tasks based on PDF documents.

        Use the following tools to answer the user's question: {tools}

        ⚠️ MANDATORY INSTRUCTIONS:
        1. You MUST use the search tool before answering ANY question
        2. After you receive search results, include ALL document sources in your answer
        3. You MUST end EVERY response with a "SOURCES:" section listing the exact document filenames 
        4. Format your response as follows:

        Your detailed answer here with information from the documents.

        SOURCES:
        - document1.pdf
        - document2.pdf

        The SOURCES section is REQUIRED. Your response will be REJECTED if you don't include it."""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = initialize_agent(
        tools=tools, 
        llm=llm, 
        agent=AgentType.OPENAI_FUNCTIONS,
        agent_kwargs={
            "memory_prompts": [prompt],
            "memory": memory
        },
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent

def extract_sources_from_documents(documents) -> List[str]:
    """Extract unique source filenames from documents"""
    sources = []
    for doc in documents:
        if doc.metadata and "source" in doc.metadata:
            source_path = doc.metadata["source"]
            source_name = Path(source_path).name
            if source_name not in sources:
                sources.append(source_name)
    return sources

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
async def chat(query: ChatQuery):
    """Send a query to chat with PDF using an agent that can access multiple document tools."""
    try:
        # Set up language model
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        llm = ChatOpenAI(temperature=0.0, model_name=model_name)
        
        # Track all tools created for this session
        all_tools = []
        
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
            
            # Create tools for specified documents
            for doc_id in query.document_ids:
                try:
                    vector_store = load_vector_store(f"data/vectorstores/{doc_id}")
                    retriever = vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3}
                    )
                    
                    tool_name = f"{doc_id[:15]}_search" if len(doc_id) > 15 else f"{doc_id}_search"
                    
                    document_tool = create_retrieval_tool(
                        retriever=retriever, 
                        name=tool_name,
                        description=f"Searches within '{doc_id}.pdf' to find relevant information."
                    )
                    
                    all_tools.append(document_tool)
                    
                except Exception as e:
                    logger.warning(f"Error with document {doc_id}: {str(e)}")
            
            if not all_tools:
                raise HTTPException(
                    status_code=500,
                    detail="Could not process any of the specified documents"
                )
        else:
            # Create tools for each indexed document
            document_dir = "data/documents"
            vectorstore_dir = "data/vectorstores"
            
            if not os.path.exists(document_dir):
                raise HTTPException(
                    status_code=404,
                    detail="No documents directory found. Please upload some PDFs first."
                )
            
            # Check for all PDF files that have corresponding vector stores
            pdf_files = []
            if os.path.exists(document_dir):
                for file in os.listdir(document_dir):
                    if file.endswith('.pdf'):
                        document_id = file.replace('.pdf', '')
                        if os.path.exists(f"{vectorstore_dir}/{document_id}"):
                            pdf_files.append(document_id)
            
            if not pdf_files:
                raise HTTPException(
                    status_code=404,
                    detail="No indexed documents found. Please index some PDFs first."
                )
            
            # Create a tool for each indexed document
            for doc_id in pdf_files:
                try:
                    vector_store = load_vector_store(f"{vectorstore_dir}/{doc_id}")
                    retriever = vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3}
                    )
                    
                    tool_name = f"{doc_id[:15]}_search" if len(doc_id) > 15 else f"{doc_id}_search"
                    
                    document_tool = create_retrieval_tool(
                        retriever=retriever, 
                        name=tool_name,
                        description=f"Searches within '{doc_id}.pdf' to find relevant information."
                    )
                    
                    all_tools.append(document_tool)
                except Exception as e:
                    logger.warning(f"Error creating tool for {doc_id}.pdf: {str(e)}")
            
            if not all_tools:
                raise HTTPException(
                    status_code=500,
                    detail="Could not create tools for any indexed documents."
                )
            
            # Log the number of document tools created
            logger.info(f"Created {len(all_tools)} document tools for agent to use")
        
        # Create a single agent with all tools
        agent = create_agent(llm, all_tools)
        
        # Track if tools were actually used
        tool_was_used = False
        retrieved_document_sources = []
        
        # Process the query with the agent, capturing sources from tool calls
        with get_openai_callback() as cb:
            class SourceTracker(BaseCallbackHandler):
                """A callback handler to track sources from tools"""
                def __init__(self):
                    super().__init__()
                
                def on_tool_start(self, serialized, input_str, **kwargs):
                    """Track when a tool is used"""
                    nonlocal tool_was_used
                    tool_was_used = True
                    logger.info(f"Tool used: {serialized.get('name', 'unknown')}")
                
                def on_tool_end(self, output, **kwargs):
                    """Track sources from tool output"""
                    try:
                        if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], dict):
                            if "sources" in output[1]:
                                nonlocal retrieved_document_sources
                                retrieved_document_sources.extend(output[1]["sources"])
                    except Exception as e:
                        logger.error(f"Error in SourceTracker.on_tool_end: {e}")
                
                # Required error handlers
                def on_llm_error(self, error, **kwargs): pass
                def on_chain_error(self, error, **kwargs): pass
                def on_tool_error(self, error, **kwargs): pass
            
            source_tracker = SourceTracker()
            
            # Invoke the agent with our callback
            agent_result = agent.invoke(
                {"input": query.query},
                {"callbacks": [source_tracker]}
            )
            logger.info(f"Agent used {cb.total_tokens} tokens")
        
        # Log the raw response
        response_text = agent_result["output"]
        logger.info(f"Tool was used: {tool_was_used}")
        
        # Save to memory
        memory.save_context(
            {"input": query.query},
            {"output": response_text}
        )
        
        # Extract sources from response
        sources = []
        main_response = response_text
        
        # Check for explicit SOURCES section
        if "SOURCES:" in response_text:
            parts = response_text.split("SOURCES:")
            main_response = parts[0].strip()
            sources_section = parts[1].strip()
            
            if not sources_section.strip():
                logger.info("Empty SOURCES section found")
            else:
                # Improved source extraction - handle "\n-" patterns and regular bullet points
                if "\n-" in sources_section or "\n- " in sources_section:
                    # Split by newline followed by dash
                    source_items = re.split(r'\n\s*-\s*', sources_section)
                    # Clean up first item which might not start with dash
                    if source_items and not source_items[0].strip().startswith("-"):
                        first_item = source_items[0].strip()
                        if first_item:
                            # Check if it has a dash at the beginning
                            if first_item.startswith("-"):
                                first_item = first_item[1:].strip()
                            source_items[0] = first_item
                    
                    # Filter out empty items and extract PDF filenames
                    pdf_pattern = r'([\w\s\-_.]+\.pdf)'
                    for item in source_items:
                        item = item.strip()
                        if item:
                            pdf_match = re.search(pdf_pattern, item)
                            if pdf_match:
                                sources.append(pdf_match.group(1))
                            elif ".pdf" in item:  # Fallback if regex fails
                                sources.append(item)
                
                # Fallback to original method if the above didn't work
                if not sources:
                    source_matches = re.findall(r'(?:^|\n)\s*[-•*]\s*([\w\s\-_.]+\.pdf)', sources_section)
                    if source_matches:
                        sources = source_matches
        
        # If no sources found in the text, use the sources we tracked from tools
        if not sources and retrieved_document_sources:
            sources = list(set(retrieved_document_sources))
            logger.info(f"Using tracked sources: {sources}")
        
        # If still no sources found but tools were used, try other extraction methods
        if not sources and tool_was_used:
            # Check for DOCUMENT SOURCES section
            if "=== DOCUMENT SOURCES" in response_text:
                source_section_match = re.search(r'=== DOCUMENT SOURCES.+?===\s*([\s\S]+?)(?:\n\n|\Z)', response_text)
                if source_section_match:
                    section_text = source_section_match.group(1)
                    source_matches = re.findall(r'[-•*]\s*([\w\s\-_.]+\.pdf)', section_text)
                    if source_matches:
                        sources = source_matches
            
            # Last resort - extract any PDF filename mentioned
            if not sources:
                all_pdf_mentions = re.findall(r'([\w\s\-_.]+\.pdf)', response_text)
                if all_pdf_mentions:
                    sources = list(set(all_pdf_mentions))
        
        # If we have explicit document IDs but no sources were found
        if not sources and query.document_ids and len(query.document_ids) > 0 and tool_was_used:
            sources = [f"{doc_id}.pdf" for doc_id in query.document_ids]
        
        # Clean up sources to ensure consistent formatting - remove any leading dashes and spaces
        cleaned_sources = []
        for source in sources:
            # Remove leading dash and space if present
            cleaned_source = source.strip()
            if cleaned_source.startswith('-'):
                cleaned_source = cleaned_source[1:].strip()
            cleaned_sources.append(cleaned_source)
        
        return ChatResponse(
            response=main_response,
            sources=cleaned_sources
        )
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-memory/", response_model=ClearMemoryResponse)
async def clear_memory():
    """Clear the conversation memory"""
    try:
        global memory
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history", 
            return_messages=True,
            k=5,
            input_key="input",
            output_key="output"
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
    """Get the conversation memory"""
    try:
        memory_vars = memory.load_memory_variables({})
        messages = memory.chat_memory.messages
        history_dicts = messages_to_dict(messages)
        
        return {"history": history_dicts}
    except Exception as e:
        logger.error(f"Error accessing memory: {str(e)}", exc_info=True)
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
