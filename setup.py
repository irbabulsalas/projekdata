#!/usr/bin/env python3
"""
Setup script for ProjekData AI Data Analysis Platform
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Requires Python 3.9+")
        return False

def install_requirements():
    """Install Python requirements"""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        f"pip install -r {requirements_file}",
        "Installing Python requirements"
    )

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    nltk_commands = [
        'python -c "import nltk; nltk.download(\'punkt\')"',
        'python -c "import nltk; nltk.download(\'stopwords\')"',
        'python -c "import nltk; nltk.download(\'wordnet\')"',
        'python -c "import nltk; nltk.download(\'vader_lexicon\')"'
    ]
    
    for cmd in nltk_commands:
        if not run_command(cmd, "Downloading NLTK data"):
            return False
    
    return True

def download_spacy_models():
    """Download spaCy models"""
    print("\n🧠 Downloading spaCy models...")
    return run_command(
        'python -m spacy download en_core_web_sm',
        "Downloading spaCy English model"
    )

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    directories = [
        "uploads",
        "exports", 
        "logs",
        "models",
        "notebooks",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created {directory} directory")
    
    return True

def check_env_file():
    """Check if .env file exists and has API key"""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    with open(env_file, 'r') as f:
        content = f.read()
        if 'GEMINI_API_KEY=' in content and 'your_gemini_api_key_here' not in content:
            print("✅ Gemini API key found in .env file")
            return True
        else:
            print("⚠️  Please update GEMINI_API_KEY in .env file")
            return False

def test_imports():
    """Test key imports"""
    print("\n🧪 Testing imports...")
    test_modules = [
        "streamlit",
        "pandas", 
        "numpy",
        "plotly",
        "sklearn",
        "nltk",
        "spacy",
        "textblob",
        "google.generativeai"
    ]
    
    failed_imports = []
    for module in test_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ All modules imported successfully")
        return True

def main():
    """Main setup function"""
    print("🚀 Setting up ProjekData AI Data Analysis Platform")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Download NLTK data
    if not download_nltk_data():
        print("❌ Failed to download NLTK data")
        sys.exit(1)
    
    # Download spaCy models
    if not download_spacy_models():
        print("❌ Failed to download spaCy models")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check environment
    check_env_file()
    
    # Test imports
    if not test_imports():
        print("❌ Some imports failed. Please check the installation")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Update your Gemini API key in .env file")
    print("2. Run the application: streamlit run app.py")
    print("3. Open your browser and go to http://localhost:8501")
    print("\n📚 For more information, check the documentation in docs/")

if __name__ == "__main__":
    main()