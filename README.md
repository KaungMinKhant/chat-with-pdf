# Chat With PDF

A web application that allows users to chat with their PDF documents using large language models.

## Project Description

Chat With PDF is an application that enables users to:
- Upload PDF documents
- Ask questions about the content of the documents
- Receive AI-generated answers based on the document content with source attribution

The application uses LangChain with OpenAI models for document processing and LLM-powered question answering.

## How It Works

The system works through the following process:

1. **Document Processing:**
   - When a PDF is uploaded, the text is extracted and split into manageable chunks
   - These chunks are embedded using OpenAI's embedding and stored in a FAISS vector database
   - Each document gets its own vector store, and a combined "all" index is maintained

2. **Question Answering:**
   - When a user asks a question, the system uses a LangChain agent with retrieval tools
   - The agent searches the vector store for relevant document chunks based on semantic similarity
   - It then uses an LLM (default: GPT-4o-mini) to generate a comprehensive answer
   - The response includes citations to the source documents

3. **Architecture:**
   - FastAPI backend provides REST endpoints for document uploading and querying
   - Vector stores are persisted to disk for subsequent sessions
   - Conversation history is maintained to support contextual follow-up questions

## Setup Instructions

### Prerequisites
- Docker and Docker Compose
- Python 3.10+
- OpenAI API key

### Running with Docker Compose

1. Clone this repository:
   ```bash
   git clone https://github.com/KaungMinKhant/chat-with-pdf.git
   cd chat-with-pdf
   ```

2. Create a `.env` file with your OpenAI API key:
   ```bash
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

3. Build and start the Docker container:
   ```bash
   docker-compose up -d
   ```

4. Access the application:
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

5. To stop the application:
   ```bash
   docker-compose down
   ```

### Local Development
1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file with your OpenAI API key
6. Run the application: `uvicorn app.main:app --reload`

## API Usage

### Upload a PDF
```bash
curl -X POST -F "file=@path/to/your/document.pdf" http://localhost:8000/upload-pdf/
```

### Ask a question about your PDF
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?"}' \
  http://localhost:8000/chat/
```

### Full API Documentation
Once the application is running, you can access the interactive API documentation at:
- http://localhost:8000/docs (Swagger UI)

## Utility Scripts

### Batch Processing PDFs

The `ingest_pdfs.py` script allows you to process multiple PDF files in batch mode, which is useful for pre-populating the system with documents:

```bash
# Process a single PDF file
python scripts/ingest_pdfs.py --file path/to/document.pdf

# Process all PDFs in a directory
python scripts/ingest_pdfs.py --directory path/to/pdf/folder

# Specify a custom output directory
python scripts/ingest_pdfs.py --directory path/to/pdf/folder --output custom/vectorstore/path

# Use a specific OpenAI API key
python scripts/ingest_pdfs.py --directory path/to/pdf/folder --api-key your-api-key
```

### Evaluating System Performance

#### Prerequisites

Before running the evaluation script, you need to install additional dependencies:

```bash
# Install evaluation-specific dependencies
pip install -r requirements_evaluation.txt
```

The `evaluate_endpoints.py` script helps evaluate the performance of the system using predetermined queries and expected answers:

```bash
# Create an example evaluation file
python scripts/evaluate_endpoints.py --create-example

# Run evaluation using an evaluation file
python scripts/evaluate_endpoints.py --input evaluation_queries.json

# Evaluate against a specific API endpoint
python scripts/evaluate_endpoints.py --input evaluation_queries.json --base-url http://your-api-url:8000

# Set a custom timeout for requests
python scripts/evaluate_endpoints.py --input evaluation_queries.json --timeout 120

# Enable debug logging
python scripts/evaluate_endpoints.py --input evaluation_queries.json --debug

# Use a custom output prefix for result files
python scripts/evaluate_endpoints.py --input evaluation_queries.json --output my_evaluation
```

The evaluation script generates detailed reports that include:
- Response times
- Text similarity scores (using embedding comparisons)
- Source attribution metrics (precision, recall, F1 score)
- Comprehensive summaries and human-readable reports

## Future Improvements

The system could be enhanced in several ways:

1. **User Interface**:
   - Develop a web frontend for easier interaction with the API
   - Add document management features (listing, deleting, categorizing)

2. **Enhanced Features**:
   - Support for more document formats (DOCX, TXT, HTML, etc.)
   - Multi-language support
   - Visual document exploration tools

3. **Alternative Models**:
   - Support for local LLMs like Llama, etc.
   - Model selection options based on user preference
