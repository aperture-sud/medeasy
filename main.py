"""
MedEasy — BART-MNLI Specialty Classifier Sidecar
Replaces the dead med-gemma loading code.

Start: uvicorn main:app --host 0.0.0.0 --port 8000
Model cache stored in ./model_cache/ (delete to free storage).
"""

import os
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Store model in project folder so it's easy to delete
os.environ.setdefault("TRANSFORMERS_CACHE", "./model_cache")
os.environ.setdefault("HF_HOME", "./model_cache")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/bart-large-mnli"

classifier = None
model_loaded = False
model_loading = False
model_load_error: Optional[str] = None


async def load_classifier():
    global classifier, model_loaded, model_loading, model_load_error
    if model_loading or model_loaded:
        return
    model_loading = True
    model_load_error = None
    try:
        logger.info(f"📥 Loading {MODEL_NAME} (this may take a few minutes on first run)...")
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            device=device,
        )
        model_loaded = True
        logger.info("✅ BART-MNLI classifier ready")
    except Exception as e:
        model_load_error = str(e)
        logger.error(f"❌ Model loading failed: {e}")
    finally:
        model_loading = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(load_classifier())
    yield


app = FastAPI(
    title="MedEasy Classifier API",
    description="BART-MNLI zero-shot specialty classifier",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    premise: str = Field(..., min_length=1, max_length=5000,
                         description="Patient symptoms / complaint text")
    hypotheses: List[str] = Field(..., min_items=1, max_items=20,
                                  description="Candidate specialty hypotheses")
    multi_label: bool = Field(default=False,
                              description="Return independent scores per label (vs. mutually exclusive)")


class ClassifyResponse(BaseModel):
    labels: List[str]
    scores: List[float]
    top_label: str
    top_score: float
    model: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_loading: bool
    model_name: str
    error: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "MedEasy BART-MNLI Classifier",
        "model": MODEL_NAME,
        "model_status": "loaded" if model_loaded else ("loading" if model_loading else "not_started"),
        "endpoints": {
            "classify": "POST /classify",
            "health": "GET /health",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if model_loaded else ("loading" if model_loading else "not_ready"),
        model_loaded=model_loaded,
        model_loading=model_loading,
        model_name=MODEL_NAME,
        error=model_load_error,
    )


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    if not model_loaded:
        detail = "Model not loaded yet"
        if model_loading:
            detail = "Model is still loading — please retry in a moment"
        elif model_load_error:
            detail = f"Model failed to load: {model_load_error}"
        raise HTTPException(status_code=503, detail=detail)

    try:
        result = classifier(
            req.premise,
            req.hypotheses,
            multi_label=req.multi_label,
        )
        return ClassifyResponse(
            labels=result["labels"],
            scores=[round(float(s), 4) for s in result["scores"]],
            top_label=result["labels"][0],
            top_score=round(float(result["scores"][0]), 4),
            model=MODEL_NAME,
        )
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.get("/ready")
async def ready():
    return {"ready": model_loaded}


@app.get("/live")
async def live():
    return {"alive": True}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info(f"🚀 Starting BART-MNLI classifier on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info", access_log=False)
