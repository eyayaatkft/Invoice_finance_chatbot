import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query
from fastapi.responses import StreamingResponse
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
import json
from typing import Literal
import shutil

# --- Load environment variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment or .env file.")
genai.configure(api_key=GEMINI_API_KEY)

# --- Gemini 2 Flash Model Name ---
GEMINI_MODEL = "models/gemini-1.5-flash-latest"

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))

app = FastAPI()

vector_store = None  # Will hold Chroma instance

TRACKING_FILE = os.path.join(os.path.dirname(__file__), "embedded_files.json")

CHAT_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "chat_history.json")

# --- Utility Functions ---
def load_document(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.pdf':
        loader = PyPDFLoader(filepath)
    elif ext == '.txt':
        loader = TextLoader(filepath)
    elif ext == '.docx':
        loader = Docx2txtLoader(filepath)
    elif ext == '.csv':
        loader = CSVLoader(filepath)
    else:
        print(f"[Skip] Unsupported file type: {filepath}")
        return []
    return loader.load()

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# --- Tracking Utilities (Unified for files and URLs) ---
def load_tracking():
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, "r") as f:
            data = json.load(f)
            if "files" not in data:
                data = {"files": data, "urls": {}}
            if "urls" not in data:
                data["urls"] = {}
            return data
    return {"files": {}, "urls": {}}

def save_tracking(tracking):
    with open(TRACKING_FILE, "w") as f:
        json.dump(tracking, f)

def get_file_mtime(filepath):
    return str(os.path.getmtime(filepath))

def get_now_timestamp():
    import time
    return str(int(time.time()))

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(history, f)

# --- ChromaDB Initialization ---
def get_vector_store():
    global vector_store
    if vector_store is None:
        embeddings = OpenAIEmbeddings()  # TODO: Replace with Gemini-compatible embeddings
        vector_store = Chroma(
            collection_name="rag_docs",
            embedding_function=embeddings,
            persist_directory=os.path.join(os.path.dirname(__file__), "chroma_db")
        )
    return vector_store

# --- Document Ingestion & Watchdog ---
class DataFolderHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            ingest_file(event.src_path)
    def on_deleted(self, event):
        if not event.is_directory:
            remove_file(event.src_path)

# --- Ingestion Logic (Files) ---
def ingest_file(filepath):
    tracking = load_tracking()
    mtime = get_file_mtime(filepath)
    if filepath in tracking["files"] and tracking["files"][filepath] == mtime:
        print(f"[Skip] Already embedded: {filepath}")
        return
    docs = load_document(filepath)
    if not docs:
        print(f"[IngestFile] No docs loaded from {filepath}")
        return
    chunks = chunk_documents(docs)
    print(f"[IngestFile] {filepath} - {len(chunks)} chunks")
    vs = get_vector_store()
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [{**chunk.metadata, "source_file": filepath} for chunk in chunks]
    print(f"[IngestFile] Metadata example: {metadatas[0] if metadatas else None}")
    vs.add_texts(texts, metadatas)
    tracking["files"][filepath] = mtime
    save_tracking(tracking)
    print(f"[Ingested] {filepath} ({len(chunks)} chunks)")

def remove_file(filepath):
    vs = get_vector_store()
    # Delete all vectors with matching source_file metadata
    deleted = vs._collection.delete(where={"source_file": filepath})
    tracking = load_tracking()
    if filepath in tracking["files"]:
        del tracking["files"][filepath]
        save_tracking(tracking)
    print(f"[Removed] {filepath} from vector store. Deleted vectors: {deleted}")

# --- Ingestion Logic (URLs) ---
def ingest_url(url):
    tracking = load_tracking()
    last_scraped = tracking["urls"].get(url)
    now = get_now_timestamp()
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[Scrape Error] {e}")
        return {"success": False, "error": f"Failed to fetch URL: {e}"}
    soup = BeautifulSoup(resp.text, "html.parser")
    for script in soup(["script", "style", "noscript"]):
        script.extract()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    page_text = "\n".join(lines)
    print(f"[Scrape] {url} - First 500 chars:\n{page_text[:500]}")
    from langchain.schema import Document
    doc = Document(page_content=page_text, metadata={"source_url": url})
    chunks = chunk_documents([doc])
    print(f"[IngestURL] {url} - {len(chunks)} chunks")
    vs = get_vector_store()
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [{**chunk.metadata, "source_url": url} for chunk in chunks]
    print(f"[IngestURL] Metadata example: {metadatas[0] if metadatas else None}")
    vs.add_texts(texts, metadatas)
    tracking["urls"][url] = now
    save_tracking(tracking)
    print(f"[Ingested URL] {url} ({len(chunks)} chunks)")
    return {"success": True, "chunks_added": len(chunks)}

def remove_url(url):
    vs = get_vector_store()
    # Delete all vectors with matching source_url metadata
    deleted = vs._collection.delete(where={"source_url": url})
    tracking = load_tracking()
    if url in tracking["urls"]:
        del tracking["urls"][url]
        save_tracking(tracking)
    print(f"[Removed URL] {url} from vector store. Deleted vectors: {deleted}")

def initial_ingest():
    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.isfile(fpath):
            ingest_file(fpath)

def start_watchdog():
    event_handler = DataFolderHandler()
    observer = Observer()
    observer.schedule(event_handler, DATA_DIR, recursive=False)
    observer_thread = threading.Thread(target=observer.start, daemon=True)
    observer_thread.start()
    print("[Watchdog] Started monitoring data folder.")

@app.on_event("startup")
def on_startup():
    global vector_store
    # TODO: Initialize Chroma vector store and embeddings
    print("[Startup] Ingesting documents...")
    initial_ingest()
    start_watchdog()

# --- Retrieval and Answer Generation ---
def retrieve_context(question, k=4):
    vs = get_vector_store()
    docs_and_scores = vs.similarity_search_with_score(question, k=k)
    docs = [doc for doc, score in docs_and_scores]
    return docs

def build_prompt(question, docs, chat_history=None):
    context = "\n\n".join([doc.page_content for doc in docs])
    history_str = ""
    if chat_history:
        for turn in chat_history[-5:]:
            history_str += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    prompt = f"""
You are a helpful assistant. Use ONLY the following context to answer the user's question. If the answer is not in the context, say you don't know.

Context:
{context}

Chat History:
{history_str}
Question: {question}
Answer as helpfully as possible:
"""
    return prompt

async def call_gemini(prompt, chat_history, temperature, max_output_tokens):
    model = genai.GenerativeModel(GEMINI_MODEL)
    # Prepare chat history for Gemini chat mode
    gemini_history = []
    for turn in chat_history[-5:]:
        gemini_history.append({"role": "user", "parts": [turn["user"]]})
        gemini_history.append({"role": "model", "parts": [turn["assistant"]]})
    chat = model.start_chat(history=gemini_history)
    response = chat.send_message(
        prompt,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_output_tokens
        }
    )
    return response.text

class ChatRequest(BaseModel):
    question: str
    temperature: float = 0.2
    max_output_tokens: int = 512

@app.post("/chat")
async def chat(request: ChatRequest):
    question = request.question
    temperature = request.temperature
    max_output_tokens = request.max_output_tokens
    chat_history = load_chat_history()
    docs = retrieve_context(question)
    prompt = build_prompt(question, docs, chat_history)
    answer = await call_gemini(prompt, chat_history, temperature, max_output_tokens)
    sources = [doc.metadata.get("source_file", "") for doc in docs]
    chat_history.append({"user": question, "assistant": answer})
    save_chat_history(chat_history)
    return {"answer": answer, "sources": sources}

@app.get("/chat/history")
def get_chat_history():
    return load_chat_history()

# --- Management Endpoints ---
@app.get("/knowledge/list")
def list_knowledge():
    tracking = load_tracking()
    return tracking

@app.post("/knowledge/remove")
def remove_knowledge(item: str = Body(...), type: Literal["file", "url"] = Body(...)):
    if type == "file":
        remove_file(item)
    elif type == "url":
        remove_url(item)
    else:
        return {"success": False, "error": "Invalid type"}
    return {"success": True}

@app.post("/knowledge/reembed")
def reembed_knowledge(item: str = Body(...), type: Literal["file", "url"] = Body(...)):
    if type == "file":
        ingest_file(item)
    elif type == "url":
        ingest_url(item)
    else:
        return {"success": False, "error": "Invalid type"}
    return {"success": True}

@app.post("/scrape-url")
async def scrape_url(url: str = Body(..., embed=True)):
    result = ingest_url(url)
    return result

@app.post("/upload")
async def upload_file(file: UploadFile):
    try:
        data_dir = DATA_DIR
        dest_path = os.path.join(data_dir, file.filename)
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"[Upload] Saved file to {dest_path}")
        ingest_file(dest_path)
        return {"success": True, "filename": file.filename}
    except Exception as e:
        print(f"[Upload Error] {e}")
        return {"success": False, "error": str(e)} 