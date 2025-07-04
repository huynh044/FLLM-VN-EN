"""
Simple FastAPI application for Vietnamese-English translation.
Pure API - only handles model loading and translation.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch
import logging
from pathlib import Path
import time

# Import translation components
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Vietnamese-English Translation API", version="1.0.0")

# CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class TranslationRequest(BaseModel):
    text: str
    max_length: Optional[int] = 128

class TranslationResponse(BaseModel):
    vietnamese: str
    english: str
    processing_time: float

class ModelStatus(BaseModel):
    model_loaded: bool
    model_path: str
    device: str

# Global translator
translator = None
current_model_path = ""

class SimpleTranslator:
    """Simple translation service."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load the fine-tuned model."""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            
            # Check if it's a PEFT model (has adapter files)
            if (Path(self.model_path) / "adapter_config.json").exists():
                # Load PEFT model
                config = PeftConfig.from_pretrained(self.model_path)
                
                # Load base model
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    config.base_model_name_or_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
                # Load PEFT adapters
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                
                logger.info("Loaded PEFT model successfully")
                
            else:
                # Load regular model
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                
                logger.info("Loaded regular model successfully")
            
            self.model.eval()
            self.model.to(self.device)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def translate(self, vietnamese_text: str, max_length: int = 128) -> dict:
        """Translate Vietnamese to English."""
        if not self.model or not self.tokenizer:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        start_time = time.time()
        
        try:
            # Format input for T5/mT5
            input_text = f"translate Vietnamese to English: {vietnamese_text}"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=256,
                truncation=True
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode result
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            processing_time = time.time() - start_time
            
            return {
                "vietnamese": vietnamese_text,
                "english": translation,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

def find_and_load_model():
    """Find and load the best available model."""
    global translator, current_model_path
    
    # Try different model paths in order of preference
    model_paths = [
        "models/fine_tuned/mt5-lora-vi-en",           # Final trained model
        "models/checkpoints/mt5-lora-vi-en/final",    # Checkpoint final
        "google/mt5-small"                            # Fallback to base model
    ]
    
    for model_path in model_paths:
        try:
            if Path(model_path).exists() or model_path.startswith("google/"):
                translator = SimpleTranslator(model_path)
                if translator.load_model():
                    current_model_path = model_path
                    logger.info(f"Successfully loaded model: {model_path}")
                    return True
        except Exception as e:
            logger.warning(f"Failed to load {model_path}: {e}")
            continue
    
    logger.error("No model could be loaded")
    return False

# Mount static files for web interface
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Serve the web interface
@app.get("/")
async def serve_web_interface():
    """Serve the main web interface."""
    return FileResponse("web/index.html")

# Startup: Load model
@app.on_event("startup")
async def startup():
    if not find_and_load_model():
        logger.error("Failed to load any model")

# API Endpoints - Pure JSON responses only
@app.get("/api/status", response_model=ModelStatus)
async def get_status():
    """Get model status."""
    return ModelStatus(
        model_loaded=translator is not None,
        model_path=current_model_path,
        device=translator.device if translator else "none"
    )

@app.get("/model-info")
async def get_model_info():
    """Get model information for web interface."""
    return {
        "model_name": current_model_path,
        "source_language": "vi",
        "target_language": "en",
        "is_loaded": translator is not None,
        "device": translator.device if translator else "none"
    }

@app.post("/api/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """Translate Vietnamese text to English."""
    if not translator:
        raise HTTPException(status_code=500, detail="Translation service not available")
    
    result = translator.translate(request.text, request.max_length)
    return TranslationResponse(**result)

@app.post("/translate")
async def translate_simple(request: TranslationRequest):
    """Simple translate endpoint for web interface."""
    if not translator:
        raise HTTPException(status_code=500, detail="Translation service not available")
    
    result = translator.translate(request.text, request.max_length)
    return {
        "original_text": result["vietnamese"],
        "translated_text": result["english"],
        "processing_time": result["processing_time"]
    }

@app.get("/api/examples")
async def get_examples():
    """Get sample Vietnamese texts."""
    return {
        "examples": [
            "Xin chào! Tôi tên là Nam.",
            "Hôm nay thời tiết rất đẹp.",
            "Tôi đang học tiếng Anh.",
            "Cảm ơn bạn rất nhiều.",
            "Tôi yêu Việt Nam.",
            "Món phở này rất ngon.",
            "Chúc bạn một ngày tốt lành!"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
