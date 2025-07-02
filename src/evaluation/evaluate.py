"""
Evaluation script for Vietnamese to English translation models.
Computes BLEU, ROUGE, and other metrics on test datasets.
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd

# Transformers and evaluation
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
from datasets import load_from_disk, Dataset
import evaluate

# PEFT for loading fine-tuned models
from peft import PeftModel, PeftConfig

# Metrics
# import sacrebleu  # Commented out due to Windows build issues
from rouge_score import rouge_scorer

# Vietnamese text processing
try:
    from underthesea import word_tokenize as vn_tokenize
except ImportError:
    print("Vietnamese NLP libraries not installed.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranslationEvaluator:
    """Comprehensive evaluation for Vietnamese-English translation models."""
    
    def __init__(self, model_path: str, tokenizer_path: str = None, use_peft: bool = False):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.use_peft = use_peft
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = self._load_model()
        
        # Setup metrics
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Setup translation pipeline
        self.translator = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
    def _load_model(self):
        """Load the fine-tuned model."""
        if self.use_peft:
            # Load PEFT model
            logger.info(f"Loading PEFT model from {self.model_path}")
            config = PeftConfig.from_pretrained(self.model_path)
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            # Load regular fine-tuned model
            logger.info(f"Loading model from {self.model_path}")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        
        model.eval()
        return model
    
    def translate_text(self, vietnamese_text: str, 
                      max_length: int = 256, 
                      num_beams: int = 4,
                      temperature: float = 1.0) -> str:
        """Translate a single Vietnamese text to English."""
        # Format input for T5-style models
        if "translate Vietnamese to English:" not in vietnamese_text:
            input_text = f"translate Vietnamese to English: {vietnamese_text}"
        else:
            input_text = vietnamese_text
        
        # Generate translation
        result = self.translator(
            input_text,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=temperature > 1.0,
            early_stopping=True
        )
        
        return result[0]['generated_text']
    
    def translate_batch(self, vietnamese_texts: List[str], 
                       batch_size: int = 8,
                       max_length: int = 256,
                       num_beams: int = 4) -> List[str]:
        """Translate a batch of Vietnamese texts."""
        translations = []
        
        for i in range(0, len(vietnamese_texts), batch_size):
            batch = vietnamese_texts[i:i + batch_size]
            
            # Format inputs
            inputs = [f"translate Vietnamese to English: {text}" for text in batch]
            
            # Tokenize
            encoded = self.tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            
            # Move to device
            if torch.cuda.is_available():
                encoded = {k: v.cuda() for k, v in encoded.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **encoded,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode
            batch_translations = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            translations.extend(batch_translations)
        
        return translations
    
    def compute_bleu_score(self, predictions: List[str], 
                          references: List[str]) -> Dict[str, float]:
        """Compute BLEU scores using different methods."""
        # SacreBLEU disabled due to Windows compatibility issues
        # sacre_bleu = sacrebleu.corpus_bleu(predictions, [references])
        
        # Hugging Face BLEU
        hf_bleu = self.bleu_metric.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )
        
        return {
            # "sacrebleu": sacre_bleu.score,
            # "sacrebleu_signature": sacre_bleu.signature,
            "hf_bleu": hf_bleu["bleu"],
            "hf_precisions": hf_bleu["precisions"]
        }
    
    def compute_rouge_scores(self, predictions: List[str], 
                           references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores."""
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
            rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
            rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)
        
        # Average scores
        return {
            "rouge1": sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"]),
            "rouge2": sum(rouge_scores["rouge2"]) / len(rouge_scores["rouge2"]),
            "rougeL": sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"])
        }
    
    def compute_length_metrics(self, predictions: List[str], 
                              references: List[str]) -> Dict[str, float]:
        """Compute length-related metrics."""
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        
        return {
            "avg_pred_length": sum(pred_lengths) / len(pred_lengths),
            "avg_ref_length": sum(ref_lengths) / len(ref_lengths),
            "length_ratio": sum(pred_lengths) / sum(ref_lengths),
            "length_diff": sum(abs(p - r) for p, r in zip(pred_lengths, ref_lengths)) / len(pred_lengths)
        }
    
    def evaluate_dataset(self, dataset_path: str, 
                        subset: str = "test",
                        max_samples: int = None,
                        output_file: str = None) -> Dict[str, Any]:
        """Evaluate model on a dataset."""
        logger.info(f"Loading dataset from {dataset_path}")
        dataset = load_from_disk(dataset_path)
        test_data = dataset[subset]
        
        if max_samples:
            test_data = test_data.select(range(min(max_samples, len(test_data))))
        
        logger.info(f"Evaluating on {len(test_data)} samples")
        
        # Extract source and target texts
        if 'vietnamese' in test_data.column_names:
            source_texts = test_data['vietnamese']
            target_texts = test_data['english']
        elif 'source' in test_data.column_names:
            source_texts = test_data['source']
            target_texts = test_data['target']
        else:
            raise ValueError("Unknown dataset format")
        
        # Generate translations
        logger.info("Generating translations...")
        predictions = self.translate_batch(source_texts)
        
        # Compute metrics
        logger.info("Computing metrics...")
        bleu_scores = self.compute_bleu_score(predictions, target_texts)
        rouge_scores = self.compute_rouge_scores(predictions, target_texts)
        length_metrics = self.compute_length_metrics(predictions, target_texts)
        
        # Combine results
        results = {
            "dataset_info": {
                "dataset_path": dataset_path,
                "subset": subset,
                "num_samples": len(test_data)
            },
            "bleu_scores": bleu_scores,
            "rouge_scores": rouge_scores,
            "length_metrics": length_metrics,
            "model_info": {
                "model_path": self.model_path,
                "use_peft": self.use_peft
            }
        }
        
        # Save detailed results if requested
        if output_file:
            detailed_results = []
            for i, (src, pred, ref) in enumerate(zip(source_texts, predictions, target_texts)):
                detailed_results.append({
                    "id": i,
                    "source": src,
                    "prediction": pred,
                    "reference": ref,
                    "pred_length": len(pred.split()),
                    "ref_length": len(ref.split())
                })
            
            # Save to file
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Save detailed results
            df = pd.DataFrame(detailed_results)
            df.to_csv(output_path.with_suffix('.csv'), index=False, encoding='utf-8')
            
            logger.info(f"Results saved to {output_path}")
        
        return results
    
    def interactive_translation(self):
        """Interactive translation session."""
        print("Vietnamese to English Translation")
        print("Type 'quit' to exit")
        print("-" * 40)
        
        while True:
            vietnamese_text = input("Vietnamese: ").strip()
            
            if vietnamese_text.lower() == 'quit':
                break
            
            if vietnamese_text:
                translation = self.translate_text(vietnamese_text)
                print(f"English: {translation}")
                print("-" * 40)
    
    def compare_models(self, other_evaluator: 'TranslationEvaluator',
                      dataset_path: str, subset: str = "test") -> Dict[str, Any]:
        """Compare two models on the same dataset."""
        # Evaluate both models
        results_1 = self.evaluate_dataset(dataset_path, subset)
        results_2 = other_evaluator.evaluate_dataset(dataset_path, subset)
        
        # Compare metrics
        comparison = {
            "model_1": {
                "path": self.model_path,
                "bleu": results_1["bleu_scores"]["hf_bleu"],
                "rouge1": results_1["rouge_scores"]["rouge1"],
                "rouge2": results_1["rouge_scores"]["rouge2"],
                "rougeL": results_1["rouge_scores"]["rougeL"]
            },
            "model_2": {
                "path": other_evaluator.model_path,
                "bleu": results_2["bleu_scores"]["hf_bleu"],
                "rouge1": results_2["rouge_scores"]["rouge1"],
                "rouge2": results_2["rouge_scores"]["rouge2"],
                "rougeL": results_2["rouge_scores"]["rougeL"]
            }
        }
        
        # Calculate improvements
        comparison["improvements"] = {
            "bleu": comparison["model_1"]["bleu"] - comparison["model_2"]["bleu"],
            "rouge1": comparison["model_1"]["rouge1"] - comparison["model_2"]["rouge1"],
            "rouge2": comparison["model_1"]["rouge2"] - comparison["model_2"]["rouge2"],
            "rougeL": comparison["model_1"]["rougeL"] - comparison["model_2"]["rougeL"]
        }
        
        return comparison


def main():
    """Main evaluation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Vietnamese-English translation model")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--dataset_path", required=True, help="Path to test dataset")
    parser.add_argument("--use_peft", action="store_true", help="Use PEFT model loading")
    parser.add_argument("--subset", default="test", help="Dataset subset to evaluate")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to evaluate")
    parser.add_argument("--output_file", help="Output file for detailed results")
    parser.add_argument("--interactive", action="store_true", help="Run interactive translation")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = TranslationEvaluator(
        model_path=args.model_path,
        use_peft=args.use_peft
    )
    
    if args.interactive:
        evaluator.interactive_translation()
    else:
        # Evaluate on dataset
        results = evaluator.evaluate_dataset(
            dataset_path=args.dataset_path,
            subset=args.subset,
            max_samples=args.max_samples,
            output_file=args.output_file
        )
        
        # Print summary
        print("\nEvaluation Results:")
        print(f"BLEU Score: {results['bleu_scores']['hf_bleu']:.2f}")
        print(f"ROUGE-1: {results['rouge_scores']['rouge1']:.3f}")
        print(f"ROUGE-2: {results['rouge_scores']['rouge2']:.3f}")
        print(f"ROUGE-L: {results['rouge_scores']['rougeL']:.3f}")
        print(f"Average prediction length: {results['length_metrics']['avg_pred_length']:.1f}")
        print(f"Length ratio: {results['length_metrics']['length_ratio']:.3f}")


if __name__ == "__main__":
    main()
