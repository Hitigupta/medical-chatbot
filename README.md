# Medical Symptom Checker

A web-based medical information assistant built with Flask and Retrieval-Augmented Generation (RAG). The application retrieves relevant information from a medical reference knowledge base stored in Pinecone and uses a language model to generate responses to user questions.

## Features

- Web interface for submitting symptom-related questions.
- PDF document ingestion and text chunking for knowledge-base preparation.
- Semantic retrieval using Hugging Face sentence embeddings.
- Pinecone vector database for storing and retrieving document embeddings.
- Groq-hosted language model for generating responses.
- Flask backend connecting the frontend, retrieval pipeline, and language model.

## Tech Stack

- **Backend:** Python, Flask
- **LLM orchestration:** LangChain
- **Language model API:** Groq API
- **Vector database:** Pinecone
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Frontend:** HTML, CSS, Bootstrap

## Architecture

1. Medical reference documents are loaded and split into smaller text chunks.
2. The chunks are converted into vector embeddings.
3. The embeddings are stored in Pinecone.
4. A user submits a question through the web interface.
5. The retriever searches the vector database for relevant context.
6. The language model generates a response using the retrieved context.
7. Flask renders the response on the webpage.

## Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd medical-chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate it using:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```dotenv
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

Never commit `.env` or expose API keys publicly.

### 5. Configure the knowledge base

The application expects a Pinecone index named `chatbot` containing vectors generated with the configured embedding model.

The app must query the same namespace in which the vectors were indexed. The medical PDF is not included in this repository; use only a copy you are authorized to process if rebuilding the knowledge base.

### 6. Run the application

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

## Deployment

The Flask application can be deployed as a web service on Render.

Configure the service's build command, start command, and environment variables using the deployment instructions in the repository or hosting dashboard.

## Medical Disclaimer

This project is intended for educational and informational purposes only. It does not provide medical diagnoses and is not a substitute for advice from a qualified healthcare professional. Seek urgent medical attention for severe or emergency symptoms.
