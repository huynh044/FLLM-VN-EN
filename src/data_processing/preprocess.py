"""
Data preprocessing module for Vietnamese to English translation dataset.
Handles cleaning, tokenization, and formatting of parallel text data from Hugging Face datasets.
"""

import json
import pandas as pd
from typing import Dict, Tuple
import re
from pathlib import Path
import logging
from tqdm import tqdm

# Vietnamese text processing
try:
    from pyvi import ViTokenizer
    from underthesea import word_tokenize as vn_tokenize
except ImportError:
    print("Vietnamese NLP libraries not installed. Install pyvi and underthesea.")

# Transformers for tokenization
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VietnameseTextProcessor:
    """Handles Vietnamese text preprocessing and cleaning."""
    
    def __init__(self):
        self.setup_patterns()
    
    def setup_patterns(self):
        """Setup regex patterns for text cleaning."""
        # Common patterns to clean
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        self.phone_pattern = re.compile(r'[\+]?[1-9]?[0-9]{7,14}')
        self.extra_spaces = re.compile(r'\s+')
        
    def clean_text(self, text: str) -> str:
        """Clean Vietnamese text."""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs, emails, phone numbers
        text = self.url_pattern.sub(' ', text)
        text = self.email_pattern.sub(' ', text)
        text = self.phone_pattern.sub(' ', text)
        
        # Remove extra whitespace
        text = self.extra_spaces.sub(' ', text)
        
        # Strip and lowercase
        text = text.strip()
        
        return text
    
    def tokenize_vietnamese(self, text: str, method: str = "pyvi") -> str:
        """Tokenize Vietnamese text."""
        try:
            if method == "pyvi":
                return ViTokenizer.tokenize(text)
            elif method == "underthesea":
                tokens = vn_tokenize(text)
                return " ".join(tokens)
            else:
                return text
        except:
            logger.warning(f"Failed to tokenize: {text[:50]}...")
            return text


class TranslationDataProcessor:
    """Main class for processing translation datasets."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.vn_processor = VietnameseTextProcessor()
        self.tokenizer = None
        
    def load_tokenizer(self, model_name: str = "google/mt5-small"):
        """Load tokenizer for the model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Loaded tokenizer: {model_name}")
    
    def load_hf_dataset(self, dataset_name: str = "ncduy/mt-en-vi") -> pd.DataFrame:
        """Load dataset from Hugging Face."""
        try:
            logger.info(f"Loading dataset from Hugging Face: {dataset_name}")
            dataset = load_dataset(dataset_name)
            
            # Sử dụng train split
            train_data = dataset['train']
            df = train_data.to_pandas()
            
            logger.info(f"Dataset columns: {df.columns.tolist()}")
            
            # Rename columns để match với preprocessing
            if 'vi' in df.columns and 'en' in df.columns:
                df = df[['vi', 'en']].copy()
                df.columns = ['vietnamese', 'english']
                logger.info(f"Renamed columns: vi -> vietnamese, en -> english")
            else:
                logger.error(f"Expected 'vi' and 'en' columns, got: {df.columns.tolist()}")
                return pd.DataFrame()
            
            logger.info(f"Loaded {len(df)} translation pairs from {dataset_name}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load HF dataset {dataset_name}: {e}")
            return pd.DataFrame()
    
    def preprocess_pair(self, vn_text: str, en_text: str) -> Tuple[str, str]:
        """Preprocess a Vietnamese-English text pair."""
        # Clean Vietnamese text
        vn_clean = self.vn_processor.clean_text(vn_text)
        
        # Clean English text (basic cleaning)
        en_clean = re.sub(r'\s+', ' ', en_text).strip()
        
        # Tokenize Vietnamese if requested
        if self.config.get('tokenize_vietnamese', False):
            vn_clean = self.vn_processor.tokenize_vietnamese(vn_clean)
        
        return vn_clean, en_clean
    
    def filter_by_length(self, df: pd.DataFrame, 
                        min_len: int = 5, max_len: int = 512) -> pd.DataFrame:
        """Filter translation pairs by length."""
        # Filter by character length
        df = df[df['vietnamese'].str.len() >= min_len]
        df = df[df['english'].str.len() >= min_len]
        df = df[df['vietnamese'].str.len() <= max_len]
        df = df[df['english'].str.len() <= max_len]
        
        # Filter by token length if tokenizer is available
        if self.tokenizer:
            def check_token_length(row):
                vn_tokens = len(self.tokenizer.encode(row['vietnamese']))
                en_tokens = len(self.tokenizer.encode(row['english']))
                return vn_tokens <= max_len and en_tokens <= max_len and \
                       vn_tokens >= min_len and en_tokens >= min_len
            
            df = df[df.apply(check_token_length, axis=1)]
        
        logger.info(f"After length filtering: {len(df)} pairs")
        return df
    
    def create_training_format(self, df: pd.DataFrame, 
                             format_type: str = "t5") -> Dataset:
        """Convert DataFrame to training format."""
        if format_type == "t5":
            # T5 format: "translate Vietnamese to English: [VN_TEXT]" -> "[EN_TEXT]"
            inputs = []
            targets = []
            
            for _, row in df.iterrows():
                input_text = f"translate Vietnamese to English: {row['vietnamese']}"
                target_text = row['english']
                
                inputs.append(input_text)
                targets.append(target_text)
            
            dataset = Dataset.from_dict({
                'input_text': inputs,
                'target_text': targets,
                'vietnamese': df['vietnamese'].tolist(),
                'english': df['english'].tolist()
            })
            
        elif format_type == "seq2seq":
            # Standard seq2seq format
            dataset = Dataset.from_dict({
                'source': df['vietnamese'].tolist(),
                'target': df['english'].tolist()
            })
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return dataset
    
    def split_dataset(self, dataset: Dataset, 
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1) -> DatasetDict:
        """Split dataset into train/validation/test sets."""
        # Shuffle dataset
        dataset = dataset.shuffle(seed=42)
        
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, total_size))
        
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        logger.info(f"Dataset split - Train: {len(train_dataset)}, "
                   f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return dataset_dict
    
    def process_hf_dataset(self, dataset_name: str = "ncduy/mt-en-vi", 
                          output_dir: str = "data/processed/mt5_format",
                          max_samples: int = None,
                          batch_size: int = 10000):
        """Process Hugging Face dataset with optional sampling and batching."""
        # Load HF dataset
        df = self.load_hf_dataset(dataset_name)
        
        if df.empty:
            logger.error("Failed to load dataset from Hugging Face")
            return
        
        # Apply sampling if specified
        if max_samples and len(df) > max_samples:
            logger.info(f"Sampling {max_samples} from {len(df)} total pairs")
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        
        logger.info(f"Processing {len(df)} pairs...")
        
        # Preprocess text pairs in batches
        logger.info("Preprocessing text pairs in batches...")
        processed_pairs = []
        
        for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
            batch_df = df.iloc[i:i+batch_size]
            
            for _, row in batch_df.iterrows():
                vn_clean, en_clean = self.preprocess_pair(row['vietnamese'], row['english'])
                if vn_clean and en_clean:  # Skip empty pairs
                    processed_pairs.append({
                        'vietnamese': vn_clean,
                        'english': en_clean
                    })
            
            # Memory cleanup every few batches
            if i % (batch_size * 5) == 0:
                import gc
                gc.collect()
        
        df_processed = pd.DataFrame(processed_pairs)
        logger.info(f"After preprocessing: {len(df_processed)} pairs")
        
        # Filter by length
        df_filtered = self.filter_by_length(df_processed)
        
        # Create training format in chunks to avoid memory issues
        logger.info("Creating training format...")
        chunk_size = min(50000, len(df_filtered))
        datasets = []
        
        for i in range(0, len(df_filtered), chunk_size):
            chunk_df = df_filtered.iloc[i:i+chunk_size]
            chunk_dataset = self.create_training_format(chunk_df, self.config.get('format_type', 't5'))
            datasets.append(chunk_dataset)
        
        # Concatenate all chunks
        if len(datasets) > 1:
            from datasets import concatenate_datasets
            full_dataset = concatenate_datasets(datasets)
        else:
            full_dataset = datasets[0]
        
        # Split dataset
        dataset_dict = self.split_dataset(full_dataset)
        
        # Save processed data
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dataset_dict.save_to_disk(str(output_path))
        logger.info(f"Saved processed dataset to {output_path}")
        
        # Save summary statistics
        stats = {
            'dataset_name': dataset_name,
            'total_original': len(df),
            'max_samples_requested': max_samples,
            'after_preprocessing': len(df_processed),
            'after_filtering': len(df_filtered),
            'train_size': len(dataset_dict['train']),
            'val_size': len(dataset_dict['validation']),
            'test_size': len(dataset_dict['test'])
        }
        
        with open(output_path / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=2)


def main():
    """Main preprocessing script with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vietnamese-English Translation Data Preprocessing")
    parser.add_argument("--dataset", type=str, default="ncduy/mt-en-vi", 
                       help="Hugging Face dataset name (default: ncduy/mt-en-vi)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (default: all)")
    parser.add_argument("--output_dir", type=str, default="data/processed/mt5_format",
                       help="Output directory (default: data/processed/mt5_format)")
    parser.add_argument("--tokenize_vn", action="store_true",
                       help="Enable Vietnamese tokenization")
    parser.add_argument("--min_length", type=int, default=5,
                       help="Minimum text length (default: 5)")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum text length (default: 256)")
    parser.add_argument("--batch_size", type=int, default=10000,
                       help="Processing batch size (default: 10000)")
    
    args = parser.parse_args()
    
    config = {
        'tokenize_vietnamese': args.tokenize_vn,
        'format_type': 't5',
        'min_length': args.min_length,
        'max_length': args.max_length
    }
    
    processor = TranslationDataProcessor(config)
    processor.load_tokenizer("google/mt5-small")
    
    # Processing information
    logger.info("🔧 Processing Configuration:")
    logger.info(f"   Dataset: {args.dataset}")
    logger.info(f"   Max samples: {args.max_samples if args.max_samples else 'ALL'}")
    logger.info(f"   Output: {args.output_dir}")
    logger.info(f"   Vietnamese tokenization: {args.tokenize_vn}")
    logger.info(f"   Length range: {args.min_length}-{args.max_length}")
    logger.info(f"   Batch size: {args.batch_size}")
    
    # Process Hugging Face dataset
    try:
        logger.info("Processing Hugging Face dataset...")
        processor.process_hf_dataset(
            dataset_name=args.dataset, 
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            batch_size=args.batch_size
        )
        logger.info("✅ Successfully processed HF dataset!")
    except Exception as e:
        logger.error(f"❌ Failed to process HF dataset: {e}")
        raise


if __name__ == "__main__":
    main()
