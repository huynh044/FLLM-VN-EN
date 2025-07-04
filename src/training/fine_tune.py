"""
Simple fine-tuning script for Vietnamese-English translation.
Uses LoRA/PEFT for efficient training.
"""

import os
import torch
import logging
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_model_and_tokenizer(model_name="google/mt5-small"):
    """Load tokenizer and model."""
    logger.info(f"Loading model: {model_name}")
    
    # Check and log device info
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with optimal dtype and safetensors
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for stability
        use_safetensors=True
    )
    
    # Setup LoRA
    lora_config = LoraConfig(
        r=16,                           # LoRA rank
        lora_alpha=32,                  # LoRA alpha
        target_modules=["q", "v"],      # Target attention layers
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Move model to device
    model = model.to(device)
    
    return model, tokenizer


def preprocess_data(examples, tokenizer, max_length=128):
    """Tokenize input and target texts."""
    # Get input and target texts
    if 'input_text' in examples:
        inputs = examples['input_text']
        targets = examples['target_text']
    else:
        raise ValueError("Dataset must have 'input_text' and 'target_text' columns")
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding=False
    )
    
    # Tokenize targets
    labels = tokenizer(
        targets,
        max_length=max_length,
        truncation=True,
        padding=False
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train_model(
    model_name="google/mt5-small",
    dataset_path="data/processed/mt5_format",
    output_dir="models/fine_tuned/mt5-lora-vi-en",
    output_checkpoint_dir="models/checkpoints/mt5-lora-vi-en",
    num_epochs=3,
    batch_size=8,
    learning_rate=3e-4
):
    """Main training function."""
    
    # 1. Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    # 2. Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    dataset = load_from_disk(dataset_path)
    
    # 3. Preprocess dataset
    logger.info("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_data(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # 4. Setup data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100
    )
    
    # 5. Training arguments with GPU optimization
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_checkpoint_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        fp16=False,  # Disable fp16 to avoid CUBLAS errors
        bf16=False,
        dataloader_pin_memory=False,  # Disable pin memory to avoid CUDA IPC errors
        dataloader_num_workers=0,  # Disable parallel loading to avoid CUDA sharing issues
        gradient_accumulation_steps=2,  # Use gradient accumulation to reduce memory
        max_grad_norm=1.0,
        report_to=[],
        remove_unused_columns=False,
        predict_with_generate=True,
        generation_max_length=128,
        save_safetensors=True,  # Use safetensors for saving
    )
    
    # 6. Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    # 7. Start training with memory optimization
    logger.info("Starting training...")
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"GPU memory before training: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    trainer.train()
    
    # Clear GPU cache after training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"GPU memory after training: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    # 8. Save final model
    final_dir = Path(output_dir)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    
    logger.info(f"Training completed! Model saved to {final_dir}")
    return trainer


if __name__ == "__main__":
    # Conservative GPU settings to avoid CUDA errors
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Use smaller batch sizes to avoid memory issues
        if gpu_memory_gb >= 16:
            batch_size = 8  # Reduced from 32
            learning_rate = 4e-4
        elif gpu_memory_gb >= 8:
            batch_size = 4  # Reduced from 16
            learning_rate = 3e-4
        elif gpu_memory_gb >= 4:
            batch_size = 2  # Reduced from 8
            learning_rate = 3e-4
        else:
            batch_size = 1  # Very small for low memory
            learning_rate = 3e-4
            
        logger.info(f"GPU detected ({gpu_memory_gb:.1f}GB) - Using conservative batch_size: {batch_size}")
        
        # CUDA environment settings for stability
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For debugging
        
    else:
        batch_size = 4
        learning_rate = 3e-4
        logger.info("CPU mode - Using conservative settings")
    
    # Train with stable configuration
    train_model(
        model_name="google/mt5-small",
        dataset_path="data/processed/mt5_format", 
        output_dir="models/fine_tuned/mt5-lora-vi-en",
        output_checkpoint_dir="models/checkpoints/mt5-lora-vi-en",
        num_epochs=3,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
