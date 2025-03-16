# Chat With PDF

A web application that allows users to chat with their PDF documents using large language models.

## Project Description

Chat With PDF is an application that enables users to:
- Upload PDF documents
- Ask questions about the content of the documents
- Receive AI-generated answers based on the document content

The application uses LangChain/LlamaIndex for document processing and connecting with language models.

## Setup Instructions

### Prerequisites
- Docker and Docker Compose
- Python 3.8 or higher (for local development)

### Running with Docker
1. Clone this repository
2. Set up your environment variables (see `.env.example` file)
3. Run `docker-compose up -d`
4. Access the application at http://localhost:8000

### Local Development
1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the application: `uvicorn app.main:app --reload`

## API Documentation

Once the application is running, you can access the interactive API documentation at:
- http://localhost:8000/docs (Swagger UI)
