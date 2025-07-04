# Vietnamese-English Translation API

Simple API server with web interface for Vietnamese to English translation.

## Quick Start

### 1. Install Dependencies
```bash
cd d:\Project\FLLM-VN-EN
pip install fastapi uvicorn transformers torch peft
```

### 2. Start Server
```bash
# Option 1: Run directly
python main.py

# Option 2: Use uvicorn
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access Interface
- **Web UI**: http://localhost:8000/
- **API Docs**: http://localhost:8000/docs

## API Usage

### Simple Translation
```bash
curl -X POST "http://localhost:8000/translate" \
     -H "Content-Type: application/json" \
     -d '{"text": "Xin chào"}'
```

**Response:**
```json
{
    "original_text": "Xin chào",
    "translated_text": "Hello",
    "processing_time": 0.5
}
```

### Advanced Translation
```bash
curl -X POST "http://localhost:8000/translate" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Xin chào thế giới",
       "max_length": 256,
       "num_beams": 4,
       "temperature": 1.0
     }'
```

### Model Info
```bash
curl -X GET "http://localhost:8000/model-info"
```

## Web Interface Features

- **Real-time Translation**: Type Vietnamese → Get English
- **Translation History**: Saves previous translations
- **Copy to Clipboard**: One-click copy results
- **Adjustable Settings**: Control translation quality
- **Responsive Design**: Works on mobile/desktop

## Configuration

### Change Model
Edit `main.py`:
```python
# Use your fine-tuned model
MODEL_PATH = "models/fine_tuned/my-model"
USE_PEFT = True  # If using LoRA adapters
```

### Translation Settings
Edit `web/static/app.js`:
```javascript
const defaultSettings = {
    maxLength: 256,    // Max output length
    numBeams: 4,       // Translation quality (1-8)
    temperature: 1.0   // Creativity (0.1-2.0)
};
```

## Using Your Fine-tuned Model

1. **Place model files:**
```
models/
├── fine_tuned/
│   └── my-model/
│       ├── adapter_config.json
│       └── adapter_model.safetensors
```

2. **Update config:**
```python
# In main.py
MODEL_PATH = "models/fine_tuned/my-model"
USE_PEFT = True
```

## Troubleshooting

### Common Issues

**Model not loading:**
- Check model path exists
- Verify all model files present

**Slow translation:**
- Reduce `num_beams` (4 → 2)
- Use smaller `max_length` (256 → 128)

**Out of memory:**
- Use CPU instead of GPU
- Reduce batch size

### Performance Tips

- **GPU**: Install CUDA-enabled PyTorch for faster translation
- **Caching**: Frequently used translations are cached
- **Batch**: Use batch endpoints for multiple texts

## Advanced Usage

### Using Python
```python
import requests

response = requests.post(
    "http://localhost:8000/translate",
    json={"text": "Xin chào Việt Nam"}
)
print(response.json()["translated_text"])
```

### Custom Parameters
```python
response = requests.post(
    "http://localhost:8000/translate",
    json={
        "text": "Tôi yêu Việt Nam",
        "max_length": 128,
        "num_beams": 2,
        "temperature": 0.7
    }
)
```

## Production Deployment

### Using Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

### Environment Variables
```bash
export MODEL_PATH="/path/to/model"
export API_PORT="8000"
export DEVICE="cuda"  # or "cpu"
```
