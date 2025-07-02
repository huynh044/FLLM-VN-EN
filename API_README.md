# Vietnamese-English Translation API & Web Interface

This directory contains the API backend and web frontend for the Vietnamese-English translation system.

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
cd d:\Project\FLLM-VN-EN
pip install fastapi uvicorn transformers torch peft
```

### 2. Start the API Server

```bash
# Start FastAPI server
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the Web Interface

Open your browser and go to:
- **Web Interface**: http://localhost:8000/
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## API Endpoints

### Translation Endpoints

#### Single Translation
```http
POST /translate
Content-Type: application/json

{
    "text": "Xin chào, tôi là một sinh viên.",
    "max_length": 256,
    "num_beams": 4,
    "temperature": 1.0
}
```

**Response:**
```json
{
    "original_text": "Xin chào, tôi là một sinh viên.",
    "translated_text": "Hello, I am a student.",
    "confidence_score": 0.95,
    "processing_time": 1.23
}
```

#### Batch Translation
```http
POST /translate-batch
Content-Type: application/json

{
    "texts": [
        "Xin chào",
        "Tôi là sinh viên",
        "Hôm nay trời đẹp"
    ],
    "max_length": 256,
    "num_beams": 4
}
```

**Response:**
```json
{
    "translations": [
        {
            "original_text": "Xin chào",
            "translated_text": "Hello",
            "processing_time": 0.5
        },
        {
            "original_text": "Tôi là sinh viên",
            "translated_text": "I am a student",
            "processing_time": 0.6
        },
        {
            "original_text": "Hôm nay trời đẹp",
            "translated_text": "The weather is nice today",
            "processing_time": 0.7
        }
    ],
    "total_processing_time": 1.8
}
```

#### Model Information
```http
GET /model-info
```

**Response:**
```json
{
    "model_name": "google/mt5-small",
    "model_type": "mT5",
    "is_loaded": true,
    "device": "cpu"
}
```

## Web Interface Features

### Main Translation Interface
- **Real-time Translation**: Type Vietnamese text and get English translation
- **Adjustable Parameters**: Control translation quality with max_length, num_beams, and temperature
- **Keyboard Shortcuts**: Press `Ctrl+Enter` to translate
- **Auto-paste Translation**: Automatically translates when you paste text (for short texts)
- **Character Counter**: Shows character count with color coding

### Translation History
- **Persistent History**: Saves your translations locally
- **Quick Reuse**: Click any history item to reload it
- **Copy Translations**: One-click copy to clipboard
- **History Management**: Clear individual items or entire history

### Advanced Features
- **Text Swapping**: Swap source and target text
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Progress Indicators**: Visual feedback during translation
- **Error Handling**: User-friendly error messages

## Configuration

### API Configuration

Edit `api.py` to configure:

```python
# Model settings
MODEL_PATH = "google/mt5-small"  # Change to your fine-tuned model
USE_PEFT = False  # Set to True if using LoRA/PEFT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# API settings
HOST = "0.0.0.0"
PORT = 8000
```

### Web Interface Configuration

Edit `web/static/app.js` to configure:

```javascript
// API endpoint
this.apiBaseUrl = 'http://localhost:8000';

// Translation settings
const defaultSettings = {
    maxLength: 256,
    numBeams: 4,
    temperature: 1.0
};
```

## Using Your Fine-tuned Model

To use your own fine-tuned model instead of the default:

1. **Update API configuration:**
```python
# In api.py
MODEL_PATH = "./models/fine_tuned/my-finetuned-model"  # Path to your model
USE_PEFT = True  # If using LoRA/PEFT adapters
```

2. **Place your model files:**
```
models/
├── my-finetuned-model/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
```

3. **For PEFT/LoRA models:**
```
models/
├── base-model/          # Base model (e.g., mt5-small)
└── peft-adapters/       # LoRA adapters
    ├── adapter_config.json
    └── adapter_model.bin
```

## Performance Optimization

### GPU Acceleration
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Model Optimization
```python
# Enable half precision (in api.py)
model = model.half()  # Reduces memory usage

# Enable model compilation (PyTorch 2.0+)
model = torch.compile(model)
```

### Caching
```python
# Enable response caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_translate(text: str):
    # Translation logic here
    pass
```

## Deployment

### Local Development
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app --bind 0.0.0.0:8000

# Using Docker (create Dockerfile)
docker build -t vn-en-translator .
docker run -p 8000:8000 vn-en-translator
```

### Environment Variables
```bash
# Set environment variables for production
export MODEL_PATH="/path/to/your/model"
export API_HOST="0.0.0.0"
export API_PORT="8000"
export WORKERS=4
```

## Troubleshooting

### Common Issues

1. **Model Loading Error**
   ```
   Solution: Check model path and ensure all model files are present
   ```

2. **CORS Error**
   ```
   Solution: Update CORS settings in api.py for your domain
   ```

3. **Out of Memory**
   ```
   Solution: Use smaller batch sizes or enable model quantization
   ```

4. **Slow Translation**
   ```
   Solution: Use GPU, reduce num_beams, or use a smaller model
   ```

### Performance Monitoring
```python
# Add logging to monitor performance
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In translation function
start_time = time.time()
# ... translation logic ...
logger.info(f"Translation took {time.time() - start_time:.2f}s")
```

## API Testing

### Using curl
```bash
# Test translation endpoint
curl -X POST "http://localhost:8000/translate" \
     -H "Content-Type: application/json" \
     -d '{"text": "Xin chào", "max_length": 256}'

# Test model info
curl -X GET "http://localhost:8000/model-info"
```

### Using Python requests
```python
import requests

# Single translation
response = requests.post(
    "http://localhost:8000/translate",
    json={"text": "Xin chào thế giới", "max_length": 256}
)
print(response.json())

# Batch translation
response = requests.post(
    "http://localhost:8000/translate-batch",
    json={"texts": ["Xin chào", "Tạm biệt"]}
)
print(response.json())
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
