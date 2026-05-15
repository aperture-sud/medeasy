from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import os
import logging
import asyncio
from contextlib import asynccontextmanager
import warnings
import sys
from dotenv import load_dotenv
load_dotenv()

os.environ["BUILD_MODE"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_NAME = "medeasy/med-gemma-finetune"
BASE_MODEL_NAME = "google/gemma-2-2b"

tokenizer = None
model = None
model_loaded = False
model_loading = False
model_load_error = None

DELAY_MODEL_LOADING = int(os.getenv("DELAY_MODEL_LOADING", "30"))

def is_build_mode():
    build_mode = os.getenv("BUILD_MODE", "false").lower() == "true"
    logger.info(f"🔧 Build mode check: BUILD_MODE={os.getenv('BUILD_MODE')}, result={build_mode}")
    return build_mode

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 App starting up...")
    if not is_build_mode():
        asyncio.create_task(delayed_model_loading())
    else:
        logger.info("📦 Build mode - skipping model loading")
    yield
    logger.info("🔄 App shutting down...")
    cleanup_model()

async def delayed_model_loading():
    logger.info(f"⏱️ Delaying model loading for {DELAY_MODEL_LOADING} seconds...")
    await asyncio.sleep(DELAY_MODEL_LOADING)
    force_load = os.getenv("FORCE_MODEL_LOADING", "false").lower() == "true"
    if force_load or not is_build_mode():
        logger.info("🚀 Starting automatic model loading...")
        await load_model_background()
    else:
        logger.info("📦 Build mode detected - skipping automatic loading")

app = FastAPI(
    title="MedEasy AI API",
    description="Medical AI API using fine-tuned med-gemma model",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=20000)
    max_length: int = Field(default=200, ge=10, le=500)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)

class ChatResponse(BaseModel):
    text: str
    model: str
    status: str
    tokens_used: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_loading: bool
    model_name: str
    build_mode: bool
    uptime: Optional[str] = None
    error: Optional[str] = None

def cleanup_model():
    global model, tokenizer, model_loaded, model_loading
    try:
        if model:
            del model
        if tokenizer:
            del tokenizer
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        import gc
        gc.collect()
        model = None
        tokenizer = None
        model_loaded = False
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "type": "server_error"})

@app.get("/")
async def root():
    return {
        "message": "MedEasy AI API - med-gemma-finetune",
        "status": "running",
        "model_status": "loaded" if model_loaded else ("loading" if model_loading else "not_started"),
        "build_mode": is_build_mode(),
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "start_loading": "/start-loading",
            "force_loading": "/force-loading"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        model_loading=model_loading,
        model_name=MODEL_NAME,
        build_mode=is_build_mode(),
        uptime=None,
        error=model_load_error
    )

async def load_model_background():
    global tokenizer, model, model_loaded, model_loading, model_load_error
    if model_loading or model_loaded:
        return
    model_loading = True
    model_load_error = None
    logger.info("🔄 Starting model loading...")
    try:
        HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not HF_TOKEN:
            raise Exception("HUGGINGFACE_HUB_TOKEN not found in environment")
        logger.info("📦 Importing libraries...")
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        import torch

        # Method 1: PEFT adapter loading
        try:
            logger.info("📥 Attempting PEFT adapter loading...")
            from peft import PeftModel
            logger.info("🔧 Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME, token=HF_TOKEN, torch_dtype=torch.float16,
                low_cpu_mem_usage=True, trust_remote_code=True
            )
            logger.info("🔧 Loading tokenizer from fine-tuned model...")
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, token=HF_TOKEN, trust_remote_code=True, use_fast=True
            )
            tokenizer_vocab_size = len(tokenizer)
            base_vocab_size = base_model.config.vocab_size
            if tokenizer_vocab_size != base_vocab_size:
                logger.info(f"📏 Resizing embeddings: {base_vocab_size} → {tokenizer_vocab_size}")
                base_model.resize_token_embeddings(tokenizer_vocab_size)
                base_model.config.vocab_size = tokenizer_vocab_size
            logger.info("🔧 Loading PEFT adapter...")
            model = PeftModel.from_pretrained(base_model, MODEL_NAME, token=HF_TOKEN)
            logger.info("✅ PEFT adapter loading successful!")

        except Exception as peft_error:
            logger.info(f"PEFT loading failed: {peft_error}")
            try:
                logger.info("📥 Attempting direct model loading...")
                tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_NAME, token=HF_TOKEN, trust_remote_code=True, use_fast=True
                )
                config = AutoConfig.from_pretrained(MODEL_NAME, token=HF_TOKEN)
                tokenizer_vocab_size = len(tokenizer)
                if hasattr(config, 'vocab_size') and config.vocab_size != tokenizer_vocab_size:
                    config.vocab_size = tokenizer_vocab_size
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        MODEL_NAME, config=config, token=HF_TOKEN,
                        torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True
                    )
                    logger.info("✅ Direct loading successful!")
                except Exception:
                    model = AutoModelForCausalLM.from_pretrained(
                        MODEL_NAME, token=HF_TOKEN, torch_dtype=torch.float16,
                        low_cpu_mem_usage=True, trust_remote_code=True, ignore_mismatched_sizes=True
                    )
                    logger.info("✅ Direct loading with ignore_mismatched_sizes successful!")
            except Exception as direct_error:
                logger.info(f"Direct loading failed: {direct_error}, falling back to base model...")
                model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_NAME, token=HF_TOKEN, torch_dtype=torch.float16,
                    low_cpu_mem_usage=True, trust_remote_code=True
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    BASE_MODEL_NAME, token=HF_TOKEN, trust_remote_code=True, use_fast=True
                )
                logger.warning("⚠️ Using base model as fallback")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        model_loaded = True
        model_loading = False
        logger.info("✅ Model loading completed!")

    except Exception as e:
        import traceback
        error_msg = f"Model loading failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        model_load_error = error_msg
        model_loading = False
        model_loaded = False
        cleanup_model()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if is_build_mode():
        raise HTTPException(status_code=503, detail="Service is in build mode.")
    if model_loading:
        raise HTTPException(status_code=503, detail="Model is loading. Please wait.")
    if not model_loaded:
        error_detail = "Model not loaded."
        if model_load_error:
            error_detail += f" Error: {model_load_error}"
        raise HTTPException(status_code=503, detail=error_detail)
    try:
        logger.info(f"Processing: {request.prompt[:80]}...")
        response_text = await asyncio.wait_for(
            generate_response(request), timeout=60.0
        )
        return ChatResponse(
            text=response_text,
            model=MODEL_NAME,
            status="success",
            tokens_used=len(tokenizer.encode(response_text)) if response_text else 0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timeout")
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Generation failed")

async def generate_response(request: ChatRequest) -> str:
    import torch
    inputs = tokenizer(
        request.prompt, return_tensors="pt", max_length=512,
        truncation=True, padding=True, return_attention_mask=True
    )
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=min(request.max_length, 200),
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response.startswith(request.prompt):
        response = response[len(request.prompt):].strip()
    return response

@app.post("/start-loading")
async def start_loading():
    if model_loaded:
        return {"message": "Model already loaded"}
    if model_loading:
        return {"message": "Model is currently loading"}
    asyncio.create_task(load_model_background())
    return {"message": "Model loading started"}

@app.post("/force-loading")
async def force_loading():
    if model_loaded:
        return {"message": "Model already loaded"}
    if model_loading:
        return {"message": "Model is currently loading"}
    asyncio.create_task(load_model_background())
    return {"message": "Force loading started"}

@app.get("/ready")
async def readiness_check():
    return {"ready": True, "model_loaded": model_loaded}

@app.get("/live")
async def liveness_check():
    return {"alive": True}

if __name__ == "__main__":
    port = int(os.environ.get("HF_PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info(f"🚀 Starting MedEasy AI server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info", access_log=False, workers=1)
