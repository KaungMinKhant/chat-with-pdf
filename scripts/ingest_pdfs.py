#!/usr/bin/env python3
"""
Script to ingest PDFs into a vector database for use with LangChain/LlamaIndex.
"""

import os
import argparse
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_single_pdf(file_path: str, output_dir: Optional[str] = None) -> None:
    """
    Process a single PDF file and create embeddings/index.
    
    Args:
        file_path: Path to the PDF file
        output_dir: Directory to save the processed outputs
    """
    logger.info(f"Processing PDF: {file_path}")
    
    # TODO: Implement PDF processing
    # 1. Load the PDF
    # 2. Split into chunks/pages
    # 3. Create embeddings
    # 4. Store in vector database
    
    logger.info(f"Finished processing: {file_path}")

def process_directory(directory_path: str, output_dir: Optional[str] = None) -> None:
    """
    Process all PDFs in a directory.
    
    Args:
        directory_path: Path to directory containing PDFs
        output_dir: Directory to save the processed outputs
    """
    logger.info(f"Processing directory: {directory_path}")
    
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {directory_path}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory_path, pdf_file)
        process_single_pdf(pdf_path, output_dir)

def main():
    parser = argparse.ArgumentParser(description="Ingest PDF documents for LLM processing")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", type=str, help="Path to a single PDF file")
    input_group.add_argument("--directory", type=str, help="Path to a directory containing PDF files")
    
    parser.add_argument("--output", type=str, help="Output directory for processed files")
    
    args = parser.parse_args()
    
    output_dir = args.output
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if args.file:
        process_single_pdf(args.file, output_dir)
    elif args.directory:
        process_directory(args.directory, output_dir)

if __name__ == "__main__":
    main()
