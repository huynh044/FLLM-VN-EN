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
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
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
    
    # 5. Training arguments
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
        fp16=torch.cuda.is_available(),
        report_to=[],  # No logging
        remove_unused_columns=False,
        predict_with_generate=True,
        generation_max_length=128,
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
    
    # 7. Start training
    logger.info("Starting training...")
    trainer.train()
    
    # 8. Save final model
    final_dir = Path(output_dir)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    
    logger.info(f"Training completed! Model saved to {final_dir}")
    return trainer


if __name__ == "__main__":
    # Simple configuration
    train_model(
        model_name="google/mt5-small",
        dataset_path="data/processed/mt5_format", 
        output_dir="models/fine_tuned/mt5-lora-vi-en",
        output_checkpoint_dir="models/checkpoints/mt5-lora-vi-en",
        num_epochs=3,
        batch_size=4,  # Small batch size for safety
        learning_rate=3e-4
    )
