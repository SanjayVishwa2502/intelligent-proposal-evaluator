# app/main.py

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import shutil
from datetime import datetime
import joblib
from sentence_transformers import SentenceTransformer
import chromadb

# --- Corrected Imports for the new structure ---
from app.src.processing.document_parser import process_new_proposal
from app.src.models.novelty_analyzer import calculate_novelty
from app.src.models.risk_analyzer import predict_risk
from app.src.processing.financial_analyzer import analyze_budget, load_rules

# --- Load all models and data ONCE at the start ---
print("--- Server is starting: Loading all models and data... ---")
FINANCIAL_RULES = load_rules('financial_rules.yaml')
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
db_client = chromadb.PersistentClient(path="vector_db")
PROPOSAL_COLLECTION = db_client.get_or_create_collection(name="proposals")
RISK_MODEL = joblib.load("trained_models/risk_model.joblib")
TFIDF_VECTORIZER = joblib.load("trained_models/tfidf_vectorizer.joblib")
print("--- All models loaded. API is ready. ---")

# --- Initialize the FastAPI App ---
app = FastAPI(title="AI R&D Proposal Evaluator")

# --- NEW: Mount static files (for CSS) and setup templates (for HTML) ---
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


# --- MODIFIED: The root endpoint now renders the HTML page ---
@app.get("/")
def home(request: Request):
    """Serves the main page with the file upload form."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/evaluate/proposal/")
async def evaluate_proposal(file: UploadFile = File(...)):
    # ... (This function's logic will be updated in our next step) ...
    return {"filename": file.filename, "status": "Evaluation logic pending."}

# This check is important for running with Uvicorn
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)