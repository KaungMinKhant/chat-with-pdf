#!/usr/bin/env python3
"""Script to ingest PDFs into a vector database for use with LangChain/LlamaIndex."""

import os
import argparse
import logging
import time
from typing import Optional
from pathlib import Path
import sys

# Add parent directory to path to import app package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv, find_dotenv

from app.utils import (
    extract_text_from_pdf, split_text, create_documents,
    create_vector_store, save_vector_store, process_pdf
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_env_vars():
    """Load environment variables from .env file."""
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / '.env'
    
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv(find_dotenv(usecwd=True))

load_env_vars()

def check_api_key():
    """Check if OpenAI API key is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables.")
        return False
    return True

def process_single_pdf(
    file_path: str, 
    output_dir: Optional[str] = None, 
    api_key: Optional[str] = None
) -> None:
    """Process a single PDF file and create embeddings/index."""
    try:
        start_time = time.time()
        _, vectorstore = process_pdf(file_path, api_key)
        
        if output_dir:
            pdf_filename = os.path.basename(file_path)
            index_name = os.path.splitext(pdf_filename)[0]
            save_path = os.path.join(output_dir, index_name)
            save_vector_store(vectorstore, save_path)

        elapsed_time = time.time() - start_time
        logger.info(f"Processed {file_path} in {elapsed_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")

def process_directory(
    directory_path: str, 
    output_dir: Optional[str] = None, 
    api_key: Optional[str] = None
) -> None:
    """Process all PDFs in a directory."""
    try:
        pdf_files = [
            file for file in os.listdir(directory_path) if file.lower().endswith('.pdf')
        ]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return
        
        all_documents = []
        successful = 0
        
        for i, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(directory_path, pdf_file)
            
            try:
                raw_text = extract_text_from_pdf(pdf_path)
                
                if not raw_text.strip():
                    continue
                
                text_chunks = split_text(raw_text)
                documents = create_documents(text_chunks, pdf_path)
                all_documents.extend(documents)
                
                vectorstore = create_vector_store(documents, api_key)
                
                if output_dir:
                    pdf_filename = os.path.basename(pdf_path)
                    index_name = os.path.splitext(pdf_filename)[0]
                    save_path = os.path.join(output_dir, index_name)
                    save_vector_store(vectorstore, save_path)
                
                successful += 1
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {str(e)}")
        
        # Create the combined vector store
        if all_documents:
            all_index_path = os.path.join(output_dir, "all")
            vectorstore = create_vector_store(all_documents, api_key)
            save_vector_store(vectorstore, all_index_path)
        
        logger.info(f"Successfully processed {successful}/{len(pdf_files)} files")
    
    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}")

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
        "--output", type=str, help="Output directory (default: data/vectorstores)"
    )
    parser.add_argument(
        "--api-key", 
        type=str, 
        help="OpenAI API key (overrides environment variable)"
    )
    
    args = parser.parse_args()
    
    api_key = args.api_key
    
    if not api_key and not check_api_key():
        return
    
    output_dir = args.output if args.output else os.path.join('data', 'vectorstores')
    os.makedirs(output_dir, exist_ok=True)
    
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
        logger.error(f"Unhandled exception: {str(e)}")
        raise

if __name__ == "__main__":
    main()
