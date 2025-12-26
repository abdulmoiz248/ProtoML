"""
Configuration file for ProtoML
Store your API keys and settings here
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys (set these in .env file)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# arXiv Configuration
ARXIV_CATEGORIES = [
    "cs.CV",  # Computer Vision
    "cs.CL",  # NLP/Computational Linguistics
    "cs.LG",  # Machine Learning
    # Add healthcare-related categories
    "q-bio.QM",  # Quantitative Methods (includes medical AI)
]

MAX_PAPERS_PER_CATEGORY = 3  # Fetch top 3 from each category to get ~10 papers total
TOTAL_PAPERS_TO_FETCH = 10

# Paper Selection Configuration
PAPERS_PER_AGENT = 5  # Split papers 5+5 between Groq and Gemini

# LLM Model Configuration
GROQ_MODEL = "llama-3.3-70b-versatile"  # or "mixtral-8x7b-32768"
GEMINI_MODEL = "gemini-2.5-flash"  # or "gemini-1.5-pro"

# Scoring Criteria Weights
SCORING_WEIGHTS = {
    "problem_relevance": 0.25,
    "dataset_quality": 0.25,
    "model_novelty": 0.25,
    "real_world_impact": 0.25
}

# Embeddings Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000  # Characters per chunk for PDF processing

# Discord Configuration
DISCORD_ENABLED = True
DISCORD_USERNAME = "ProtoML"

# Output Directories
OUTPUT_DIR = "output"
PDF_CACHE_DIR = os.path.join(OUTPUT_DIR, "pdf_cache")
EMBEDDINGS_DIR = os.path.join(OUTPUT_DIR, "embeddings")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PDF_CACHE_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
