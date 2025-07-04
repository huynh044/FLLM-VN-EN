"""
Inference script for Vietnamese to English translation.
Provides easy-to-use interface for translation tasks.
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
import argparse

# Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)

# PEFT for loading fine-tuned models
from peft import PeftModel, PeftConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VietnameseTranslator:
    """Easy-to-use Vietnamese to English translator."""
    
    def __init__(self, 
                 model_path: str,
                 tokenizer_path: Optional[str] = None,
                 use_peft: bool = False,
                 device: str = "auto"):
        """
        Initialize the translator.
        
        Args:
            model_path: Path to the fine-tuned model
            tokenizer_path: Path to tokenizer (defaults to model_path)
            use_peft: Whether the model uses PEFT (LoRA)
            device: Device to run on ('auto', 'cpu', 'cuda')
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.use_peft = use_peft
        
        # Set device
        if device == "auto":
            self.device = 0 if torch.cuda.is_available() else -1
        elif device == "cuda":
            self.device = 0
        else:
            self.device = -1
        
        # Load model and tokenizer
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        
        # Create translation pipeline
        self.translator = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        logger.info(f"Translator initialized with model: {model_path}")
    
    def _load_tokenizer(self):
        """Load tokenizer."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            logger.info(f"Loaded tokenizer from {self.tokenizer_path}")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _load_model(self):
        """Load the fine-tuned model with PyTorch 2.5.1 compatibility."""
        # Try multiple loading strategies for PyTorch 2.5.1 compatibility
        loading_strategies = [
            ("safetensors", {"use_safetensors": True}),
            ("without_safetensors", {"use_safetensors": False}),
            ("trust_remote_code", {"trust_remote_code": True}),
            ("basic", {})
        ]
        
        for strategy_name, kwargs in loading_strategies:
            try:
                logger.info(f"Trying {strategy_name} loading strategy...")
                
                if self.use_peft:
                    # Load PEFT model
                    logger.info(f"Loading PEFT model from {self.model_path}")
                    config = PeftConfig.from_pretrained(self.model_path)
                    base_model = AutoModelForSeq2SeqLM.from_pretrained(
                        config.base_model_name_or_path,
                        torch_dtype=torch.float32,  # Use float32 for stability
                        **kwargs
                    )
                    model = PeftModel.from_pretrained(
                        base_model, 
                        self.model_path,
                        **kwargs
                    )
                else:
                    # Load regular fine-tuned model
                    logger.info(f"Loading model from {self.model_path}")
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float32,  # Use float32 for stability
                        **kwargs
                    )
                
                model.eval()
                logger.info(f"Successfully loaded model using {strategy_name} strategy")
                return model
                
            except Exception as e:
                logger.warning(f"Failed to load model with {strategy_name} strategy: {e}")
                continue
        
        # If all strategies fail, try one more time with minimal settings
        logger.error("All loading strategies failed. Trying minimal fallback...")
        try:
            if self.use_peft:
                config = PeftConfig.from_pretrained(self.model_path)
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    config.base_model_name_or_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                model = PeftModel.from_pretrained(
                    base_model, 
                    self.model_path,
                    low_cpu_mem_usage=True
                )
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            
            model.eval()
            logger.warning("Loaded model with minimal fallback - functionality may be limited")
            return model
            
        except Exception as final_error:
            logger.error(f"All loading attempts failed: {final_error}")
            logger.error("This is likely due to PyTorch 2.5.1 compatibility issues.")
            logger.error("Consider upgrading to PyTorch 2.6.0+ or converting model to safetensors format.")
            raise
    
    def translate(self, 
                  text: Union[str, List[str]],
                  max_length: int = 256,
                  num_beams: int = 4,
                  temperature: float = 1.0,
                  do_sample: bool = False,
                  top_p: float = 0.9,
                  repetition_penalty: float = 1.1) -> Union[str, List[str]]:
        """
        Translate Vietnamese text to English.
        
        Args:
            text: Vietnamese text(s) to translate
            max_length: Maximum length of generated translation
            num_beams: Number of beams for beam search
            temperature: Sampling temperature (>1.0 for more random)
            do_sample: Whether to use sampling instead of beam search
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repeating tokens
            
        Returns:
            Translated text(s)
        """
        # Handle single string
        if isinstance(text, str):
            return self._translate_single(
                text, max_length, num_beams, temperature, 
                do_sample, top_p, repetition_penalty
            )
        
        # Handle list of strings
        elif isinstance(text, list):
            return self._translate_batch(
                text, max_length, num_beams, temperature,
                do_sample, top_p, repetition_penalty
            )
        
        else:
            raise ValueError("Input must be string or list of strings")
    
    def _translate_single(self, text: str, max_length: int, num_beams: int,
                         temperature: float, do_sample: bool, top_p: float,
                         repetition_penalty: float) -> str:
        """Translate a single text."""
        # Format input for T5-style models
        if not text.startswith("translate Vietnamese to English:"):
            input_text = f"translate Vietnamese to English: {text}"
        else:
            input_text = text
        
        # Generate translation
        try:
            result = self.translator(
                input_text,
                max_length=max_length,
                num_beams=num_beams if not do_sample else 1,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p if do_sample else None,
                repetition_penalty=repetition_penalty,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            return result[0]['generated_text'].strip()
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return f"[Translation Error: {str(e)}]"
    
    def _translate_batch(self, texts: List[str], max_length: int, num_beams: int,
                        temperature: float, do_sample: bool, top_p: float,
                        repetition_penalty: float) -> List[str]:
        """Translate a batch of texts."""
        results = []
        
        # Process in batches to avoid memory issues
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch:
                translation = self._translate_single(
                    text, max_length, num_beams, temperature,
                    do_sample, top_p, repetition_penalty
                )
                batch_results.append(translation)
            
            results.extend(batch_results)
        
        return results
    
    def translate_file(self, 
                      input_file: str,
                      output_file: str,
                      input_column: str = "vietnamese",
                      output_column: str = "english_translation",
                      **kwargs):
        """
        Translate texts from a file.
        
        Args:
            input_file: Path to input file (CSV, JSON, or TXT)
            output_file: Path to output file
            input_column: Column name for Vietnamese text (for CSV/JSON)
            output_column: Column name for English translation
            **kwargs: Additional translation parameters
        """
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        # Read input file
        if input_path.suffix.lower() == '.csv':
            import pandas as pd
            df = pd.read_csv(input_file)
            texts = df[input_column].tolist()
            
        elif input_path.suffix.lower() == '.json':
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                texts = [item[input_column] for item in data]
            else:
                texts = [data[input_column]]
                
        elif input_path.suffix.lower() == '.txt':
            with open(input_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
                
        else:
            raise ValueError("Unsupported input file format. Use CSV, JSON, or TXT.")
        
        # Translate
        logger.info(f"Translating {len(texts)} texts...")
        translations = self.translate(texts, **kwargs)
        
        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() == '.csv':
            import pandas as pd
            if input_path.suffix.lower() == '.csv':
                # Add translation column to existing DataFrame
                df[output_column] = translations
                df.to_csv(output_file, index=False, encoding='utf-8')
            else:
                # Create new DataFrame
                df = pd.DataFrame({
                    input_column: texts,
                    output_column: translations
                })
                df.to_csv(output_file, index=False, encoding='utf-8')
                
        elif output_path.suffix.lower() == '.json':
            results = []
            for text, translation in zip(texts, translations):
                results.append({
                    input_column: text,
                    output_column: translation
                })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
        elif output_path.suffix.lower() == '.txt':
            with open(output_file, 'w', encoding='utf-8') as f:
                for translation in translations:
                    f.write(translation + '\n')
        
        logger.info(f"Translations saved to {output_file}")
    
    def interactive_mode(self):
        """Run interactive translation mode."""
        print("Vietnamese to English Translator")
        print("=" * 50)
        print("Commands:")
        print("  'quit' or 'exit' - Exit the translator")
        print("  'help' - Show this help message")
        print("  'settings' - Adjust translation settings")
        print("=" * 50)
        
        # Default settings
        settings = {
            'max_length': 256,
            'num_beams': 4,
            'temperature': 1.0,
            'do_sample': False,
            'repetition_penalty': 1.1
        }
        
        while True:
            try:
                vietnamese_text = input("\nVietnamese: ").strip()
                
                if vietnamese_text.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                elif vietnamese_text.lower() == 'help':
                    print("Commands:")
                    print("  'quit' or 'exit' - Exit the translator")
                    print("  'help' - Show this help message")
                    print("  'settings' - Adjust translation settings")
                    continue
                
                elif vietnamese_text.lower() == 'settings':
                    print("\nCurrent settings:")
                    for key, value in settings.items():
                        print(f"  {key}: {value}")
                    
                    print("\nEnter new values (press Enter to keep current):")
                    for key in settings:
                        new_value = input(f"  {key} [{settings[key]}]: ").strip()
                        if new_value:
                            try:
                                if key in ['max_length', 'num_beams']:
                                    settings[key] = int(new_value)
                                elif key in ['temperature', 'repetition_penalty']:
                                    settings[key] = float(new_value)
                                elif key == 'do_sample':
                                    settings[key] = new_value.lower() in ['true', '1', 'yes']
                            except ValueError:
                                print(f"Invalid value for {key}, keeping current value")
                    continue
                
                elif vietnamese_text:
                    # Translate
                    translation = self.translate(vietnamese_text, **settings)
                    print(f"English: {translation}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Vietnamese to English Translator")
    
    # Model arguments
    parser.add_argument("--model_path", required=True, 
                       help="Path to fine-tuned model")
    parser.add_argument("--tokenizer_path", 
                       help="Path to tokenizer (defaults to model_path)")
    parser.add_argument("--use_peft", action="store_true",
                       help="Use PEFT model loading")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                       help="Device to run on")
    
    # Translation arguments
    parser.add_argument("--text", help="Text to translate")
    parser.add_argument("--input_file", help="Input file to translate")
    parser.add_argument("--output_file", help="Output file for translations")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    
    # Generation parameters
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum translation length")
    parser.add_argument("--num_beams", type=int, default=4,
                       help="Number of beams for beam search")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--do_sample", action="store_true",
                       help="Use sampling instead of beam search")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                       help="Repetition penalty")
    
    args = parser.parse_args()
    
    # Initialize translator
    translator = VietnameseTranslator(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        use_peft=args.use_peft,
        device=args.device
    )
    
    # Translation parameters
    translate_kwargs = {
        'max_length': args.max_length,
        'num_beams': args.num_beams,
        'temperature': args.temperature,
        'do_sample': args.do_sample,
        'repetition_penalty': args.repetition_penalty
    }
    
    # Run based on mode
    if args.interactive:
        translator.interactive_mode()
    
    elif args.text:
        translation = translator.translate(args.text, **translate_kwargs)
        print(f"Vietnamese: {args.text}")
        print(f"English: {translation}")
    
    elif args.input_file and args.output_file:
        translator.translate_file(
            args.input_file, 
            args.output_file,
            **translate_kwargs
        )
    
    else:
        print("Please specify --text, --input_file with --output_file, or --interactive")


if __name__ == "__main__":
    main()
