#!/usr/bin/env python3
"""Evaluation script for Chat With PDF endpoints with embedding-based comparison."""

import argparse
import json
import time
import sys
import os
import requests
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import statistics
from pathlib import Path
import logging
from datetime import datetime
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv, find_dotenv

# Add parent directory to path to import app package if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_env_vars():
    """Load environment variables from .env file."""
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / '.env'
    
    if env_path.exists():
        logger.info(f"Loading .env from {env_path}")
        load_dotenv(env_path)
    else:
        logger.info("Looking for .env file in other locations")
        load_dotenv(find_dotenv(usecwd=True))
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables.")
    else:
        logger.info("OPENAI_API_KEY found successfully.")

class ChatPdfEvaluator:
    """Evaluates the Chat With PDF application endpoints using embedding-based comparison."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 60,
        report_dir: str = "evaluation_reports",
        openai_api_key: Optional[str] = None
    ):
        self.base_url = base_url.rstrip('/')  # Remove trailing slash if present
        self.timeout = timeout
        self.report_dir = report_dir
        self.chat_endpoint = f"{self.base_url}/chat/"
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            logger.error("No OpenAI API key provided. Embeddings will not work!")
        
        # Initialize embeddings model
        self.embeddings_model = None
        try:
            from langchain_openai import OpenAIEmbeddings
            self.embeddings_model = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            logger.info("OpenAI embeddings model initialized successfully")
        except ImportError:
            logger.error("langchain_openai not available. Install with 'pip install langchain-openai'")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings model: {e}")
        
        os.makedirs(report_dir, exist_ok=True)
        
        # Test connection to API
        try:
            requests.get(f"{self.base_url}/", timeout=5)
            logger.info(f"Successfully connected to API at {self.base_url}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not connect to API at {self.base_url}: {e}")
    
    def load_evaluation_data(self, filepath: str) -> List[Dict]:
        """Load evaluation data from JSON or CSV file."""
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                return json.load(f)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            return df.to_dict('records')
        else:
            raise ValueError("Evaluation file must be JSON or CSV")
    
    def query_endpoint(self, endpoint: str, query: str, document_ids: Optional[List[str]] = None) -> Tuple[Dict, float]:
        """Send query to endpoint and measure response time."""
        start_time = time.time()
        payload = {"query": query}
        if document_ids:
            payload["document_ids"] = document_ids
            
        # Debug info
        logger.debug(f"Sending request to {endpoint}")
        logger.debug(f"Payload: {payload}")
            
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                endpoint, 
                json=payload, 
                headers=headers,
                timeout=self.timeout
            )
            response_time = time.time() - start_time
            
            # Log response status and headers for debugging
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            
            if response.status_code != 200:
                logger.error(f"Endpoint {endpoint} returned status {response.status_code}")
                logger.error(f"Response content: {response.text}")
                return {"error": f"HTTP {response.status_code}: {response.text}"}, response_time
                
            response.raise_for_status()
            return response.json(), response_time
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to {endpoint} failed: {e}")
            return {"error": str(e)}, time.time() - start_time
    
    def compute_embedding_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between embeddings of two texts."""
        if not self.embeddings_model or not text1 or not text2:
            return 0.0
            
        try:
            # Get embeddings
            embedding1 = self.embeddings_model.embed_query(text1)
            embedding2 = self.embeddings_model.embed_query(text2)
            
            # Compute cosine similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing embedding similarity: {e}")
            return 0.0
    
    def compare_answers(self, actual_answer: str, expected_answer: Optional[str]) -> Dict[str, float]:
        """Compare actual answer with expected answer using embedding similarity."""
        if not expected_answer or not actual_answer:
            return {
                "embedding_similarity": None,
            }
        
        # Semantic similarity using embeddings
        embedding_similarity = self.compute_embedding_similarity(actual_answer, expected_answer)
        
        return {
            "embedding_similarity": embedding_similarity
        }
    
    def compare_sources(self, actual_sources: List[str], expected_sources: Optional[List[str]]) -> Dict[str, float]:
        """
        Compare returned sources with expected sources.
        Returns precision, recall, f1 score and other metrics.
        """
        if not expected_sources:
            return {
                "precision": None,
                "recall": None,
                "f1": None,
                "all_expected_found": None
            }
        
        # Process actual sources to handle combined entries
        processed_actual = []
        for source in actual_sources:
            # Check if this source entry contains multiple sources (with newlines and bullet points)
            if isinstance(source, str) and '\n' in source:
                # Split by newline and extract each filename
                split_sources = re.findall(r'[-•*]?\s*([\w\s\-_.]+\.pdf)', source)
                processed_actual.extend([s.strip() for s in split_sources if s.strip()])
            else:
                processed_actual.append(source.strip() if isinstance(source, str) else source)
        
        # Remove duplicates
        actual_set = set(processed_actual)
        expected_set = set(expected_sources)
        
        # Calculate precision, recall, and F1
        true_positives = len(actual_set.intersection(expected_set))
        
        precision = true_positives / len(actual_set) if actual_set else 0.0
        recall = true_positives / len(expected_set) if expected_set else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Check if all expected sources were found
        all_found = expected_set.issubset(actual_set)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "all_expected_found": all_found
        }
    
    def run_evaluation(self, evaluation_data: List[Dict]) -> Dict:
        """Run evaluation on all queries."""
        results = {
            "chat": []
        }
        
        for i, item in enumerate(evaluation_data):
            query = item["query"]
            document_ids = item.get("document_ids")
            expected_answer = item.get("expected_answer")
            expected_sources = item.get("expected_sources")
            
            logger.info(f"Processing query {i+1}/{len(evaluation_data)}: {query[:50]}...")
            
            # Evaluate chat endpoint
            chat_response, chat_time = self.query_endpoint(self.chat_endpoint, query, document_ids)
            chat_response_text = chat_response.get("response", "")
            chat_metrics = self.compare_answers(chat_response_text, expected_answer)
            chat_source_metrics = self.compare_sources(
                chat_response.get("sources", []), expected_sources
            )
            
            # Store results
            results["chat"].append({
                "query": query,
                "document_ids": document_ids,
                "response_time": chat_time,
                "response": chat_response_text,
                "sources": chat_response.get("sources", []),
                "expected_answer": expected_answer,
                "expected_sources": expected_sources,
                **chat_metrics,
                **{f"source_{k}": v for k, v in chat_source_metrics.items()}
            })
            
            # Log progress
            logger.info(f"  Chat: {chat_time:.2f}s")
            
            if expected_answer:
                logger.info(f"  Chat embedding sim: {chat_metrics['embedding_similarity']:.4f}")
            
            # Log source metrics if expected sources were provided
            if expected_sources:
                logger.info(f"  Chat source F1: {chat_source_metrics['f1']:.2f}")
        
        return results
    
    def generate_summary(self, results: Dict) -> Dict:
        """Generate summary metrics from evaluation results."""
        summary = {}
        
        for endpoint, data in results.items():
            # Collect metrics
            response_times = [item["response_time"] for item in data]
            
            # Text similarity metrics
            embedding_similarities = [item["embedding_similarity"] for item in data if item["embedding_similarity"] is not None]
            
            # Source metrics
            source_precisions = [item["source_precision"] for item in data if item["source_precision"] is not None]
            source_recalls = [item["source_recall"] for item in data if item["source_recall"] is not None]
            source_f1s = [item["source_f1"] for item in data if item["source_f1"] is not None]
            all_sources_found = [item["source_all_expected_found"] for item in data if item["source_all_expected_found"] is not None]
            
            # Helper function to safely calculate mean
            def safe_mean(values):
                return statistics.mean(values) if values else 0
            
            # Helper function to safely calculate stdev
            def safe_stdev(values):
                return statistics.stdev(values) if len(values) > 1 else 0
            
            summary[endpoint] = {
                "count": len(data),
                "response_time": {
                    "mean": safe_mean(response_times),
                    "median": statistics.median(response_times) if response_times else 0,
                    "min": min(response_times) if response_times else 0,
                    "max": max(response_times) if response_times else 0,
                    "stdev": safe_stdev(response_times)
                },
                "text_similarity": {
                    "embedding": safe_mean(embedding_similarities) if embedding_similarities else None,
                },
                "source_attribution": {
                    "precision": safe_mean(source_precisions) if source_precisions else None,
                    "recall": safe_mean(source_recalls) if source_recalls else None,
                    "f1": safe_mean(source_f1s) if source_f1s else None,
                    "all_found_rate": safe_mean([int(v) for v in all_sources_found]) if all_sources_found else None
                }
            }
        
        return summary
    
    def save_results(self, results: Dict, summary: Dict, output_prefix: str = "evaluation"):
        """Save the detailed results and summary to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_path = os.path.join(self.report_dir, f"{output_prefix}_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        summary_path = os.path.join(self.report_dir, f"{output_prefix}_summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create a human-readable report
        report_path = os.path.join(self.report_dir, f"{output_prefix}_report_{timestamp}.txt")
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("CHAT WITH PDF EVALUATION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            for endpoint, metrics in summary.items():
                f.write(f"ENDPOINT: {endpoint}\n")
                f.write("=" * 40 + "\n")
                
                f.write(f"Queries processed: {metrics['count']}\n\n")
                
                # Response time section
                rt = metrics["response_time"]
                f.write("RESPONSE TIME (seconds)\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Mean:   {rt['mean']:.2f}\n")
                f.write(f"  Median: {rt['median']:.2f}\n")
                f.write(f"  Min:    {rt['min']:.2f}\n")
                f.write(f"  Max:    {rt['max']:.2f}\n")
                f.write(f"  StDev:  {rt['stdev']:.2f}\n\n")
                
                # Text similarity section
                ts = metrics["text_similarity"]
                if ts["embedding"] is not None:
                    f.write("TEXT SIMILARITY SCORES\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"  Embedding Similarity: {ts['embedding']:.4f}\n\n")
                
                # Source attribution section
                sa = metrics["source_attribution"]
                if any(v is not None for v in sa.values()):
                    f.write("SOURCE ATTRIBUTION METRICS\n")
                    f.write("-" * 40 + "\n")
                    if sa["precision"] is not None:
                        f.write(f"  Precision:     {sa['precision']:.4f}\n")
                    if sa["recall"] is not None:
                        f.write(f"  Recall:        {sa['recall']:.4f}\n")
                    if sa["f1"] is not None:
                        f.write(f"  F1 Score:      {sa['f1']:.4f}\n")
                    if sa["all_found_rate"] is not None:
                        f.write(f"  All Sources Found Rate: {sa['all_found_rate']:.2%}\n")
                    f.write("\n")
                
                f.write("\n" + "=" * 60 + "\n\n")
        
        logger.info(f"Saved detailed results to {results_path}")
        logger.info(f"Saved summary metrics to {summary_path}")
        logger.info(f"Saved human-readable report to {report_path}")
        
        return results_path, summary_path, report_path

def create_example_evaluation_file(output_path: str):
    """Create an example evaluation file with sample queries."""
    example_data = [
        {
            "query": "What are the key evaluation metrics for text-to-SQL systems?",
            "document_ids": ["Rajkumar_et_al_2022_Evaluating_the_Text_to_SQL"],
            "expected_answer": "The key evaluation metrics include percentage of valid SQL predictions, execution accuracy, and test-suite execution accuracy.",
            "expected_sources": ["Rajkumar_et_al_2022_Evaluating_the_Text_to_SQL.pdf"]
        },
        {
            "query": "Summarize the challenges in prompt engineering for LLMs",
            "document_ids": ["Chang_and_Fosler_Lussier_2023_How_to_Prompt"],
            "expected_answer": "",  # Empty string means we're not checking the answer
            "expected_sources": ["Chang_and_Fosler_Lussier_2023_How_to_Prompt.pdf"]
        }
    ]
    
    with open(output_path, 'w') as f:
        json.dump(example_data, f, indent=2)
    
    logger.info(f"Created example evaluation file at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Chat With PDF endpoints")
    parser.add_argument("--input", "-i", type=str, help="Input file with evaluation queries (JSON or CSV)")
    parser.add_argument("--create-example", "-c", action="store_true", help="Create an example evaluation file")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
    parser.add_argument("--output", "-o", type=str, default="evaluation", help="Output file prefix")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--openai-api-key", type=str, help="OpenAI API key (overrides environment variable)")
    
    args = parser.parse_args()
    
    # Configure debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    
    # Load environment variables
    load_env_vars()
    
    if args.create_example:
        create_example_evaluation_file("evaluation_queries.json")
        return
    
    if not args.input:
        parser.error("Please provide an input file (--input) or use --create-example")
    
    evaluator = ChatPdfEvaluator(
        base_url=args.base_url, 
        timeout=args.timeout,
        openai_api_key=args.openai_api_key
    )
    
    # Check if embeddings model is properly initialized
    if not evaluator.embeddings_model:
        logger.error("Embeddings model not initialized. Results will be limited.")
    
    try:
        evaluation_data = evaluator.load_evaluation_data(args.input)
        results = evaluator.run_evaluation(evaluation_data)
        summary = evaluator.generate_summary(results)
        evaluator.save_results(results, summary, args.output)
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
