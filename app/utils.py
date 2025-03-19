"""Utility functions for PDF processing and vector store operations."""

import os
import logging
from pathlib import Path
import PyPDF2
from typing import List, Optional

from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)

            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
                
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise

def split_text(text: str) -> List[str]:
    """Split text into chunks using LangChain's CharacterTextSplitter."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_documents(text_chunks: List[str], source_path: str) -> List[Document]:
    """Create LangChain Document objects from text chunks."""
    documents = [
        Document(
            page_content=chunk,
            metadata={"source": source_path, "chunk_id": i}
        )
        for i, chunk in enumerate(text_chunks)
    ]
    return documents

def create_vector_store(documents: List[Document], api_key: Optional[str] = None):
    """Create a FAISS vector store from documents."""
    try:
        embeddings_kwargs = {}
        if api_key:
            embeddings_kwargs["openai_api_key"] = api_key
            
        embeddings = OpenAIEmbeddings(**embeddings_kwargs)
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store
    except Exception as e:
        logger.error(f"Failed to create FAISS vector store: {str(e)}")
        raise

def load_vector_store(directory: str, api_key: Optional[str] = None):
    """Load a FAISS vector store from a directory."""
    try:
        embeddings_kwargs = {}
        if api_key:
            embeddings_kwargs["openai_api_key"] = api_key
            
        embeddings = OpenAIEmbeddings(**embeddings_kwargs)
        vector_store = FAISS.load_local(
            directory, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        raise

def save_vector_store(vector_store, directory: str):
    """Save a FAISS vector store to a directory."""
    try:
        os.makedirs(os.path.dirname(directory), exist_ok=True)
        vector_store.save_local(directory)
    except Exception as e:
        logger.error(f"Error saving vector store: {str(e)}")
        raise

def process_pdf(file_path: str, api_key: Optional[str] = None) -> (List[Document], FAISS):
    """Process a PDF file and create documents and vector store."""
    raw_text = extract_text_from_pdf(file_path)
    text_chunks = split_text(raw_text)
    documents = create_documents(text_chunks, file_path)
    vector_store = create_vector_store(documents, api_key)
    return documents, vector_store

def update_all_vector_store(
    documents: List[Document], 
    all_index_path: str,
    api_key: Optional[str] = None
):
    """Update or create the combined 'all' vector store."""
    try:
        if os.path.exists(all_index_path):
            try:
                all_vectorstore = load_vector_store(all_index_path, api_key)
                all_vectorstore.add_documents(documents)
            except Exception:
                all_vectorstore = create_vector_store(documents, api_key)
        else:
            all_vectorstore = create_vector_store(documents, api_key)
            
        save_vector_store(all_vectorstore, all_index_path)
    except Exception as e:
        logger.error(f"Failed to update 'all' vector store: {str(e)}")
        raise
