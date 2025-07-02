#!/usr/bin/env python3
"""
Quick start example for Vietnamese-English translation model.
This demonstrates basic usage of the fine-tuned model.
"""

def quick_start_demo():
    """Quick demonstration of the translation model."""
    print("🇻🇳 ➡️ 🇺🇸 Vietnamese to English Translation Demo")
    print("=" * 50)
    
    # Sample Vietnamese sentences to translate
    test_sentences = [
        "Xin chào! Tôi tên là Nam.",
        "Hôm nay thời tiết rất đẹp.",
        "Tôi đang học tiếng Anh.",
        "Cảm ơn bạn rất nhiều.",
        "Tôi yêu Việt Nam."
    ]
    
    print("📝 Sample Vietnamese sentences to translate:")
    for i, sentence in enumerate(test_sentences, 1):
        print(f"{i}. {sentence}")
    
    print("\n🔧 To get started:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the Jupyter notebook: notebooks/vietnamese_english_translation.ipynb")
    print("3. Or use the training pipeline: python scripts/train_pipeline.py")
    
    print("\n📚 Project structure:")
    print("├── src/")
    print("│   ├── data_processing/    # Data preprocessing")
    print("│   ├── training/           # Model training")
    print("│   ├── evaluation/         # Model evaluation")
    print("│   └── inference/          # Translation inference")
    print("├── notebooks/              # Jupyter notebooks")
    print("├── configs/                # Configuration files")
    print("├── scripts/                # Utility scripts")
    print("└── data/                   # Dataset storage")


def check_environment():
    """Check if the environment is set up correctly."""
    print("\n🔍 Checking environment...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not installed")
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")
    
    try:
        import datasets
        print(f"✅ Datasets: {datasets.__version__}")
    except ImportError:
        print("❌ Datasets not installed")


if __name__ == "__main__":
    quick_start_demo()
    check_environment()
    
    print("\n🚀 Ready to start fine-tuning your Vietnamese-English translation model!")
    print("📖 Check the README.md for detailed instructions.")