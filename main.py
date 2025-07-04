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
    """Simple translation service compatible with PyTorch 2.5.1+cu121."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        
        # Log PyTorch version and device info
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
    def load_model(self):
        """Load the fine-tuned model with PyTorch 2.5.1 compatibility."""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            
            # Check if it's a PEFT model (has adapter files)
            if (Path(self.model_path) / "adapter_config.json").exists():
                logger.info("Detected PEFT model - loading with adapters...")
                return self._load_peft_model()
            else:
                logger.info("Loading regular fine-tuned model...")
                return self._load_regular_model()
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _load_peft_model(self):
        """Load PEFT model with multiple fallback strategies."""
        strategies = [
            ("safetensors", {"use_safetensors": True}),
            ("without_safetensors", {"use_safetensors": False}),
            ("trust_remote_code", {"trust_remote_code": True}),
            ("basic", {})
        ]
        
        for strategy_name, kwargs in strategies:
            try:
                logger.info(f"Trying PEFT loading with {strategy_name} strategy...")
                
                # Load PEFT config
                config = PeftConfig.from_pretrained(self.model_path)
                
                # Load base model with current strategy
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    config.base_model_name_or_path,
                    torch_dtype=torch.float32,  # Use float32 for PyTorch 2.5.1 stability
                    **kwargs
                )
                
                # Load PEFT adapters
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                
                self.model.eval()
                self.model.to(self.device)
                
                logger.info(f"Successfully loaded PEFT model with {strategy_name} strategy")
                return True
                
            except Exception as e:
                logger.warning(f"PEFT loading failed with {strategy_name}: {e}")
                continue
        
        logger.error("All PEFT loading strategies failed")
        return False
    
    def _load_regular_model(self):
        """Load regular model with multiple fallback strategies."""
        strategies = [
            ("safetensors", {"use_safetensors": True}),
            ("without_safetensors", {"use_safetensors": False}),
            ("trust_remote_code", {"trust_remote_code": True}),
            ("basic", {})
        ]
        
        for strategy_name, kwargs in strategies:
            try:
                logger.info(f"Trying regular loading with {strategy_name} strategy...")
                
                # Load model
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,  # Use float32 for PyTorch 2.5.1 stability
                    **kwargs
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                
                self.model.eval()
                self.model.to(self.device)
                
                logger.info(f"Successfully loaded regular model with {strategy_name} strategy")
                return True
                
            except Exception as e:
                logger.warning(f"Regular loading failed with {strategy_name}: {e}")
                continue
        
        logger.error("All regular loading strategies failed")
        return False
    
    def translate(self, vietnamese_text: str, max_length: int = 128) -> dict:
        """Translate Vietnamese to English with PyTorch 2.5.1 optimizations."""
        if not self.model or not self.tokenizer:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        start_time = time.time()
        
        try:
            # Format input for T5/mT5
            input_text = f"translate Vietnamese to English: {vietnamese_text}"
            
            # Tokenize with proper handling
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=256,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate translation with stable settings for PyTorch 2.5.1
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode result
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            processing_time = time.time() - start_time
            
            # Clean up GPU memory if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                "vietnamese": vietnamese_text,
                "english": translation,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            # Clean up GPU memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

def find_and_load_model():
    """Find and load the best available model with PyTorch 2.5.1 compatibility."""
    global translator, current_model_path
    
    # Try different model paths in order of preference
    model_paths = [
        "models/fine_tuned/mt5-lora-vi-en",           # Final trained model
        "models/checkpoints/mt5-lora-vi-en/final",    # Checkpoint final
        "models/mt5-small-vi-en",                     # Alternative path
        "google/mt5-small"                            # Fallback to base model
    ]
    
    for model_path in model_paths:
        try:
            logger.info(f"Attempting to load model: {model_path}")
            
            # Check if local path exists or it's a HuggingFace model
            if Path(model_path).exists() or model_path.startswith("google/"):
                translator = SimpleTranslator(model_path)
                if translator.load_model():
                    current_model_path = model_path
                    logger.info(f"Successfully loaded model: {model_path}")
                    
                    # Log model info
                    logger.info(f"Model device: {translator.device}")
                    logger.info(f"Model type: {'PEFT' if (Path(model_path) / 'adapter_config.json').exists() else 'Regular'}")
                    
                    return True
                else:
                    logger.warning(f"Failed to load model: {model_path}")
                    
        except Exception as e:
            logger.warning(f"Failed to load {model_path}: {e}")
            continue
    
    logger.error("No model could be loaded - all loading attempts failed")
    logger.error("Please check:")
    logger.error("1. Model files exist and are not corrupted")
    logger.error("2. PyTorch and transformers versions are compatible")
    logger.error("3. Sufficient memory is available")
    return False

# Mount static files for web interface
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Serve the web interface
@app.get("/")
async def serve_web_interface():
    """Serve the main web interface."""
    return FileResponse("web/index.html")

# Startup: Load model with detailed logging
@app.on_event("startup")
async def startup():
    logger.info("Starting FLLM-VN-EN Translation API...")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    if not find_and_load_model():
        logger.error("Failed to load any model - API will run with limited functionality")
        logger.error("Please train a model first or check model paths")
    else:
        logger.info("API startup complete - ready to serve translations")

# API Endpoints - Pure JSON responses only
@app.get("/api/status", response_model=ModelStatus)
async def get_status():
    """Get detailed model status."""
    device_info = "none"
    if translator and translator.device:
        device_info = translator.device
        if translator.device == "cuda" and torch.cuda.is_available():
            device_info += f" ({torch.cuda.get_device_name(0)})"
    
    return ModelStatus(
        model_loaded=translator is not None,
        model_path=current_model_path,
        device=device_info
    )

@app.get("/model-info")
async def get_model_info():
    """Get detailed model information for web interface."""
    device_info = "none"
    model_type = "unknown"
    
    if translator:
        device_info = translator.device
        if translator.device == "cuda" and torch.cuda.is_available():
            device_info += f" ({torch.cuda.get_device_name(0)})"
        
        # Determine model type
        if current_model_path and (Path(current_model_path) / "adapter_config.json").exists():
            model_type = "PEFT/LoRA"
        elif current_model_path and current_model_path.startswith("google/"):
            model_type = "Base Model"
        else:
            model_type = "Fine-tuned"
    
    return {
        "model_name": current_model_path,
        "model_type": model_type,
        "source_language": "vi",
        "target_language": "en",
        "is_loaded": translator is not None,
        "device": device_info,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/api/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """Translate Vietnamese text to English with enhanced error handling."""
    if not translator:
        raise HTTPException(
            status_code=500, 
            detail="Translation service not available - please check if model is loaded"
        )
    
    # Validate input
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    
    if len(request.text) > 1000:
        raise HTTPException(status_code=400, detail="Input text too long (max 1000 characters)")
    
    try:
        result = translator.translate(request.text.strip(), request.max_length)
        return TranslationResponse(**result)
    except Exception as e:
        logger.error(f"Translation API error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/translate")
async def translate_simple(request: TranslationRequest):
    """Simple translate endpoint for web interface with enhanced error handling."""
    if not translator:
        raise HTTPException(
            status_code=500, 
            detail="Translation service not available - please check if model is loaded"
        )
    
    # Validate input
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    
    if len(request.text) > 1000:
        raise HTTPException(status_code=400, detail="Input text too long (max 1000 characters)")
    
    try:
        result = translator.translate(request.text.strip(), request.max_length)
        return {
            "original_text": result["vietnamese"],
            "translated_text": result["english"],
            "processing_time": result["processing_time"]
        }
    except Exception as e:
        logger.error(f"Translation web API error: {e}")
        return {
            "original_text": request.text,
            "translated_text": f"[Translation Error: {str(e)}]",
            "processing_time": 0.0
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
    
    # Log startup info
    logger.info("Starting FLLM-VN-EN Translation Server...")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    # Start server
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
