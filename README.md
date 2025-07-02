# Fine-tuning LLM for Vietnamese to English Translation

This project focuses on fine-tuning a large language model (LLM) for Vietnamese to English translation tasks.

## Project Structure

```
FLLM-VN-EN/
├── data/
│   ├── processed/           # Preprocessed and formatted data
│   └── validation/          # Validation datasets
├── models/
│   ├── base/               # Base model downloads
│   ├── checkpoints/        # Training checkpoints
│   └── fine_tuned/         # Final fine-tuned models
├── src/
│   ├── data_processing/    # Data preprocessing scripts
│   ├── training/           # Training and fine-tuning scripts
│   ├── evaluation/         # Model evaluation scripts
│   └── inference/          # Inference and prediction scripts
├── notebooks/              # Jupyter notebooks for experimentation
├── configs/                # Configuration files
├── scripts/                # Utility scripts
└── requirements.txt        # Python dependencies
```

## Setup

1. Create conda environment with Python 3.11:
   ```bash
   conda create -n fllm-vn-en python=3.11
   conda activate fllm-vn-en
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

1. Prepare your Vietnamese-English translation dataset
2. Run data preprocessing: `python src/data_processing/preprocess.py`
3. Start fine-tuning: `python src/training/fine_tune.py`
4. Evaluate the model: `python src/evaluation/evaluate.py`

## Models Supported

- GPT-2/GPT-3.5 variants
- T5 (Text-to-Text Transfer Transformer)
- mT5 (multilingual T5)
- BART
- MarianMT
- Custom transformer architectures

## Dataset Sources
https://huggingface.co/datasets/ncduy/mt-en-vi

python -m pip freeze --local > requirements.txt // Save all packages
