import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# 1. Configuration & Global Variables
# --------------------------------------------------------------------------

# --- Model Configuration ---
# This is the path to the directory where you saved your fine-tuned model
MODEL_PATH = "./sentinel-bert-base"  # Change this to your model path

# --- Business Logic Configuration ---
# IMPORTANT: Set this to the best threshold you found from your script
OPTIMAL_THRESHOLD = 0.35 # Example value, change this!

# Define the labels in the same order as your training
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Chunking configuration
MAX_TOKENS = 512
CHUNK_OVERLAP = 50  # Overlap tokens between chunks to maintain context



# --------------------------------------------------------------------------
# 2. Load Model & Tokenizer
# --------------------------------------------------------------------------

# Check if the model path exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model directory not found at: {MODEL_PATH}")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load the fine-tuned model using the 'pipeline' for easy inference
# We specify the device to use GPU if available, otherwise CPU
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    return_all_scores=True, # Get scores for all labels, not just the top one
    device=device
)

# --------------------------------------------------------------------------
# 3. Initialize FastAPI App & Define Data Models
# --------------------------------------------------------------------------
app = FastAPI()

class ModerateRequest(BaseModel):
    text: str

class ModerateResponse(BaseModel):
    scores: dict[str, float]
    final_decision: str
    flagged_categories: list[str]


# --------------------------------------------------------------------------
# 4. Helper Functions
# --------------------------------------------------------------------------

def chunk_text_by_tokens(text: str, max_tokens: int = MAX_TOKENS, overlap: int = CHUNK_OVERLAP):
    """
    Split text into chunks based on token count with overlap.
    
    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk
        overlap: Number of overlapping tokens between chunks
    
    Returns:
        List of text chunks
    """
    # Tokenize the full text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # If text fits in one chunk, return as is
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    stride = max_tokens - overlap
    
    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i:i + max_tokens]
        
        # Decode tokens back to text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        
        # If we've covered all tokens, break
        if i + max_tokens >= len(tokens):
            break
    
    logger.info(f"Text split into {len(chunks)} chunks (original tokens: {len(tokens)})")
    return chunks


def get_chunk_predictions(chunk: str):
    """
    Get predictions for a single chunk of text.
    
    Args:
        chunk: Text chunk to analyze
    
    Returns:
        Dictionary of label probabilities
    """
    results = classifier(chunk, truncation=True, max_length=MAX_TOKENS)
    
    probabilities = {}
    
    for item in results[0]:
        label_key = item['label']
        
        if label_key.startswith('LABEL_'):
            label_idx = int(label_key.split('_')[1])
            if label_idx < len(LABELS):
                label_name = LABELS[label_idx]
                probability = item['score']
                probabilities[label_name] = probability
    
    return probabilities


def aggregate_chunk_predictions(chunk_predictions: list):
    """
    Aggregate predictions from multiple chunks by taking the maximum probability for each label.
    
    Args:
        chunk_predictions: List of dictionaries containing label probabilities
    
    Returns:
        Dictionary with aggregated probabilities (max for each label)
    """
    if not chunk_predictions:
        return {label: 0.0 for label in LABELS}
    
    # Initialize with zeros
    aggregated = {label: 0.0 for label in LABELS}
    
    # Take maximum probability across all chunks for each label
    for chunk_pred in chunk_predictions:
        for label in LABELS:
            if label in chunk_pred:
                aggregated[label] = max(aggregated[label], chunk_pred[label])
    
    logger.info(f"Aggregated predictions from {len(chunk_predictions)} chunks")
    for label, prob in aggregated.items():
        logger.info(f"  {label}: max_probability={prob:.6f}")
    
    return aggregated


# --------------------------------------------------------------------------
# 5. API Endpoint
# --------------------------------------------------------------------------
@app.post("/v1/moderate", response_model=ModerateResponse)
async def moderate_text(request: ModerateRequest):
    
    # Handle empty input text
    if not request.text.strip():
        return {
            "scores": {label: 0.0 for label in LABELS}, 
            "final_decision": "ACCEPT",
            "flagged_categories": []
        }

    logger.info(f"Processing text: '{request.text[:100]}{'...' if len(request.text) > 100 else ''}'")
    
    # Split text into chunks if it exceeds token limit
    chunks = chunk_text_by_tokens(request.text)
    
    # Get predictions for each chunk
    chunk_predictions = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)} (length: {len(chunk)} chars)")
        chunk_pred = get_chunk_predictions(chunk)
        chunk_predictions.append(chunk_pred)
        logger.info(f"Chunk {i+1} predictions: {chunk_pred}")
    
    # Aggregate predictions by taking max probability for each label
    aggregated_probabilities = aggregate_chunk_predictions(chunk_predictions)
    
    # Convert probabilities to binary scores using threshold
    final_scores = {}
    for label in LABELS:
        probability = aggregated_probabilities[label]
        binary_score = 1 if probability > OPTIMAL_THRESHOLD else 0
        final_scores[label] = binary_score
        logger.info(f"{label}: probability={probability:.6f}, binary_score={binary_score} (threshold={OPTIMAL_THRESHOLD})")

    # Log the processed scores
    logger.info(f"Final scores: {final_scores}")

    # Implement the final decision logic - REJECT if any label is 1
    flagged_categories = [label for label, score in final_scores.items() if score == 1]
    
    # REJECT if even one label is flagged
    decision = "REJECT" if flagged_categories else "ACCEPT"
    
    logger.info(f"Decision: {decision}, Flagged categories: {flagged_categories}")
    
    return {
        "scores": final_scores, 
        "final_decision": decision,
        "flagged_categories": flagged_categories
    }