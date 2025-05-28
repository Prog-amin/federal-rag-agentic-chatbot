#!/usr/bin/env python3
"""
Setup script to prepare the environment for the RAG Agentic System
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def check_mysql():
    """Check if MySQL is available"""
    try:
        result = subprocess.run(['mysql', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ MySQL detected: {result.stdout.strip()}")
        else:
            print("⚠️  MySQL not found in PATH")
    except FileNotFoundError:
        print("⚠️  MySQL not found. Please install MySQL server.")

def check_ollama():
    """Check if Ollama is available"""
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama detected: {result.stdout.strip()}")
            return True
        else:
            print("⚠️  Ollama not responding")
            return False
    except FileNotFoundError:
        print("⚠️  Ollama not found. Please install Ollama for local LLM.")
        return False

def install_requirements():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)

def setup_env_file():
    """Create .env file from template"""
    env_template = Path('.env.template')
    env_file = Path('.env')
    
    if env_template.exists() and not env_file.exists():
        shutil.copy(env_template, env_file)
        print("✅ Created .env file from template")
        print("📝 Please edit .env file with your configuration")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️  .env.template not found")

def create_directories():
    """Create necessary directories"""
    directories = ['logs', 'data', 'data/raw', 'data/processed']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Created necessary directories")

def check_ollama_model():
    """Check if required Ollama model is available"""
    if not check_ollama():
        return False
    
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True)
        
        if 'qwen2.5:0.5b' in result.stdout or 'qwen2.5:1b' in result.stdout:
            print("✅ Qwen2.5 model found")
            return True
        else:
            print("⚠️  Qwen2.5 model not found")
            print("💡 Run: ollama pull qwen2.5:0.5b")
            return False
            
    except subprocess.CalledProcessError:
        print("⚠️  Could not check Ollama models")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up RAG Agentic System Environment")
    print("=" * 50)
    
    # Check system requirements
    check_python_version()
    check_mysql()
    
    # Setup Python environment
    install_requirements()
    
    # Setup configuration
    setup_env_file()
    create_directories()
    
    # Check LLM setup
    check_ollama_model()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your MySQL credentials")
    print("2. Start MySQL server")
    print("3. Install Ollama model: ollama pull qwen2.5:0.5b")
    print("4. Run data pipeline: python run_pipeline.py")
    print("5. Start the application: python main.py")

if __name__ == "__main__":
    main()