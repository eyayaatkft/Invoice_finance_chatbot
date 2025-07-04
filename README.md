# Invoice Finance Assistant: RAG Chatbot System

## Overview
A full-stack Retrieval-Augmented Generation (RAG) chatbot that answers user questions based on documents in a local folder, using Gemini 2 Flash as the LLM. The system features a Next.js + Tailwind CSS frontend and a FastAPI backend with LangChain, ChromaDB, and Gemini API integration.

---

## Project Structure

```
Invoice_Finance_Asistant/
│
├── backend/         # FastAPI app, LangChain, ChromaDB, Gemini API
├── frontend/        # Next.js + Tailwind CSS chat UI
├── data/            # Knowledge base (PDF, DOCX, TXT, CSV, etc.)
└── README.md
```

---

## Setup Instructions

### 1. Backend (FastAPI)
- Go to `backend/`
- Install dependencies (Python 3.9+ recommended):
  ```bash
  pip install -r requirements.txt
  ```
- Start the FastAPI server:
  ```bash
  uvicorn main:app --reload
  ```

### 2. Frontend (Next.js)
- Go to `frontend/`
- Install dependencies:
  ```bash
  npm install
  # or
  yarn install
  ```
- Start the development server:
  ```bash
  npm run dev
  # or
  yarn dev
  ```

### 3. Data Folder
- Place your knowledge base documents (PDF, DOCX, TXT, CSV, etc.) in the `data/` folder.

---

## Features
- RAG chatbot with Gemini 2 Flash LLM
- Document ingestion and retrieval (ChromaDB + LangChain)
- Real-time chat UI with streaming answers
- Modular, extensible architecture

---

## Future Extensions
- User authentication
- Feedback loop
- More file types and advanced document parsing 