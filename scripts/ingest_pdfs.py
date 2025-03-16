#!/usr/bin/env python3
"""
Script to ingest PDFs into a vector database for use with LangChain/LlamaIndex.
"""

import os
import argparse
from typing import List, Optional, Dict
import logging
import time
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv, find_dotenv

# PDF processing
import PyPDF2

# LangChain imports
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Try to load the .env file from different possible locations
def load_env_vars():
    """Load environment variables from .env file."""
    # Try project root
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / '.env'
    
    if env_path.exists():
        logger.info(f"Loading .env from {env_path}")
        load_dotenv(env_path)
    else:
        # Try current directory
        env_found = load_dotenv(find_dotenv(usecwd=True))
        if env_found:
            logger.info("Loaded .env file from current working directory")
        else:
            logger.warning(
                "No .env file found. Make sure OPENAI_API_KEY is set in environment."
            )

load_env_vars()


# Check if API key is available
def check_api_key():
    """Check if OpenAI API key is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables.")
        logger.error("Please set it using:")
        logger.error(
            "  1. Create a .env file in the project root with "
            "OPENAI_API_KEY='your-api-key'"
        )
        logger.error(
            "  2. Or export OPENAI_API_KEY='your-api-key' in your terminal"
            " 3. Or pass the API key as an argument to the script"
            " using the --api-key flag."
        )
        return False
    return True


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
        
    Raises:
        Exception: If there are issues reading the PDF
    """
    logger.info(f"Extracting text from: {file_path}")
    text = ""

    try:
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            logger.info(f"PDF has {num_pages} pages")

            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
                
        logger.info(f"Successfully extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise


def create_faiss_vector_store(documents, embeddings):
    """
    Create a FAISS vector store from documents and embeddings.
    
    Args:
        documents: List of langchain Document objects
        embeddings: Embedding model to use
        
    Returns:
        FAISS vector store instance
        
    Raises:
        ImportError: If FAISS cannot be initialized
    """
    try:
        logger.info("Creating FAISS vector store")
        vector_store = FAISS.from_documents(documents, embeddings)
        logger.info("Successfully created FAISS vector store")
        return vector_store
    except Exception as e:
        error_msg = (
            "Failed to create FAISS vector store. If using FAISS with NumPy 2.x, "
            "try: pip uninstall numpy faiss-cpu -y && pip install numpy<2.0.0 "
            "faiss-cpu"
        )
        logger.error(f"{error_msg}. Error: {str(e)}")
        raise ImportError(f"{error_msg}. Error: {str(e)}")


def process_single_pdf(
    file_path: str, 
    output_dir: Optional[str] = None, 
    api_key: Optional[str] = None
) -> None:
    """
    Process a single PDF file and create embeddings/index.
    
    Args:
        file_path: Path to the PDF file
        output_dir: Directory to save the processed outputs
        api_key: Optional API key to override environment variable
    """
    try:
        start_time = time.time()
        logger.info(f"Processing PDF: {file_path}")

        # 1. Load and extract text from the PDF
        raw_text = extract_text_from_pdf(file_path)
        if not raw_text.strip():
            logger.warning(f"No text content extracted from {file_path}")
            return

        # 2. Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_text(raw_text)
        logger.info(f"Split into {len(texts)} text chunks")

        # 3. Create documents with metadata
        documents = [
            Document(
                page_content=text_chunk,
                metadata={"source": file_path, "chunk_id": i}
            )
            for i, text_chunk in enumerate(texts)
        ]

        # 4. Create embeddings with OpenAI API key
        embeddings_kwargs = {}
        if api_key:
            embeddings_kwargs["openai_api_key"] = api_key

        embeddings = OpenAIEmbeddings(**embeddings_kwargs)

        # 5. Store in FAISS vector database 
        vectorstore = create_faiss_vector_store(documents, embeddings)

        # 6. Save the FAISS index to disk
        if output_dir:
            # Create a filename based on the original PDF name
            pdf_filename = os.path.basename(file_path)
            index_name = os.path.splitext(pdf_filename)[0]
            save_path = os.path.join(output_dir, index_name)

            vectorstore.save_local(save_path)
            logger.info(f"Saved FAISS index to {save_path}")

        elapsed_time = time.time() - start_time
        logger.info(
            f"Finished processing: {file_path} in {elapsed_time:.2f} seconds"
        )

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)


def process_directory(
    directory_path: str, 
    output_dir: Optional[str] = None, 
    api_key: Optional[str] = None
) -> None:
    """
    Process all PDFs in a directory.
    
    Args:
        directory_path: Path to directory containing PDFs
        output_dir: Directory to save the processed outputs
        api_key: Optional API key to override environment variable
    """
    logger.info(f"Processing directory: {directory_path}")
    
    try:
        pdf_files = [
            f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')
        ]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        successful = 0
        for i, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(directory_path, pdf_file)
            logger.info(f"Processing file {i+1}/{len(pdf_files)}: {pdf_file}")
            
            try:
                process_single_pdf(pdf_path, output_dir, api_key)
                successful += 1
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {str(e)}")
                
        logger.info(
            f"Directory processing complete. Successfully processed "
            f"{successful}/{len(pdf_files)} files."
        )
    
    except Exception as e:
        logger.error(
            f"Error processing directory {directory_path}: {str(e)}", 
            exc_info=True
        )


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents for LLM processing"
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--file", type=str, help="Path to a single PDF file"
    )
    input_group.add_argument(
        "--directory", type=str, help="Path to a directory containing PDF files"
    )

    parser.add_argument(
        "--output", type=str, help="Output directory for processed files (default: data/vectorstores)"
    )
    parser.add_argument(
        "--api-key", 
        type=str, 
        help="OpenAI API key (overrides environment variable)"
    )
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key
    
    # If no API key provided as argument, check environment
    if not api_key and not check_api_key():
        return
    
    # Set default output directory to data/vectorstores if not provided
    output_dir = args.output if args.output else os.path.join('data', 'vectorstores')
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output will be saved to: {output_dir}")
    
    try:
        if args.file:
            if not os.path.isfile(args.file):
                logger.error(f"File not found: {args.file}")
                return
            process_single_pdf(args.file, output_dir, api_key)
        elif args.directory:
            if not os.path.isdir(args.directory):
                logger.error(f"Directory not found: {args.directory}")
                return
            process_directory(args.directory, output_dir, api_key)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
