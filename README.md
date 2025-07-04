# Fine-tuning LLM for Vietnamese to English Translation

This project focuses on fine-tuning a large language model (LLM) for Vietnamese to English translation tasks.

## Cuda Version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Jan_15_19:38:46_Pacific_Standard_Time_2025
Cuda compilation tools, release 12.8, V12.8.61
Build cuda_12.8.r12.8/compiler.35404655_0

## Project Structure

```
FLLM-VN-EN/
├── data/
│   ├── processed/           # Preprocessed and formatted data
├── models/
│   ├── base/               # Base model downloads
│   ├── checkpoints/        # Training checkpoints
│   └── fine_tuned/         # Final fine-tuned models
├── src/
│   ├── data_processing/    # Data preprocessing scripts
│   ├── training/           # Training and fine-tuning scripts
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
   python -m pip install -r requirements.txt
   ```

## Quick Start

1. Prepare your Vietnamese-English translation dataset
2. Run data preprocessing: `python src/data_processing/preprocess.py`
3. Start fine-tuning: `python src/training/fine_tune.py`
4. Evaluate the model: `python src/evaluation/evaluate.py`

## Models Supported
- mT5 (multilingual T5)

## Dataset Sources
https://huggingface.co/datasets/ncduy/mt-en-vi
   ```bash
   python -m pip freeze --local > requirements.txt // Save all packages
   ```
## Simple Steps
- Step 1: Download and preprocess dataset: python src/data_preprocessing/preprocess.py
- Step 2: Fine-tune the dataset: python src/training/fine_tune.py
- After that: you can use the model trained and start script: python main.py
--> http://localhost:8000
