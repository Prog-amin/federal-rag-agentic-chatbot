import os
from dotenv import load_dotenv

load_dotenv()

# Database Configuration (SQLite for HF Spaces)
DB_CONFIG = {
    'type': 'sqlite',
    'db_path': 'federal_registry.db'
}

# LLM Configuration (Using Hugging Face Inference API)
LLM_CONFIG = {
    'base_url': 'https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium',
    'model': os.getenv('HF_MODEL', 'microsoft/DialoGPT-medium'),
    'api_key': os.getenv('HUGGINGFACE_API_TOKEN', ''),
    'use_openai_format': False,  # HF API has different format
    'fallback_model': 'microsoft/DialoGPT-medium'
}

# Alternative: If you have OpenAI API key, you can use GPT models
OPENAI_CONFIG = {
    'base_url': 'https://api.openai.com/v1',
    'model': 'gpt-3.5-turbo',
    'api_key': os.getenv('OPENAI_API_KEY', ''),
    'use_openai_format': True
}

# Use OpenAI if API key is available, otherwise use HF
USE_OPENAI = bool(os.getenv('OPENAI_API_KEY'))
ACTIVE_LLM_CONFIG = OPENAI_CONFIG if USE_OPENAI else LLM_CONFIG

# Federal Registry API Configuration
FEDERAL_REGISTRY_API = {
    'base_url': 'https://www.federalregister.gov/api/v1',
    'documents_endpoint': '/documents.json',
    'rate_limit_delay': 1.0  # seconds between requests
}

# Data Pipeline Configuration
PIPELINE_CONFIG = {
    'data_dir': '/tmp/data',  # Use /tmp for HF Spaces
    'raw_data_dir': '/tmp/data/raw',
    'processed_data_dir': '/tmp/data/processed',
    'retention_days': 7,
    'max_pages_per_run': 10,  # Limit to avoid timeout
    'batch_size': 50,
    'request_timeout': 30
}

# Gradio Configuration
GRADIO_CONFIG = {
    'server_name': '0.0.0.0',
    'server_port': 7860,
    'show_error': True,
    'debug': False,
    'share': False,
    'enable_queue': True,
    'max_threads': 4
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S'
}

# HF Spaces specific settings
HF_SPACES_CONFIG = {
    'max_memory_mb': 16 * 1024,  # 16GB limit
    'max_cpu_cores': 8,
    'timeout_seconds': 120,
    'enable_persistence': True,  # Keep SQLite file
    'temp_dir': '/tmp'
}

def get_hf_token():
    """Get Hugging Face token from environment"""
    token = os.getenv('HUGGINGFACE_API_TOKEN') or os.getenv('HF_TOKEN')
    if not token:
        print("⚠️  Warning: No Hugging Face token found. Some features may not work.")
    return token

def validate_config():
    """Validate configuration settings"""
    issues = []
    
    # Check HF token
    if not get_hf_token() and not USE_OPENAI:
        issues.append("No Hugging Face API token found")
    
    # Check OpenAI key if using OpenAI
    if USE_OPENAI and not OPENAI_CONFIG['api_key']:
        issues.append("OpenAI API key not found but USE_OPENAI is True")
    
    if issues:
        print("⚠️  Configuration issues:")
        for issue in issues:
            print(f"   - {issue}")
    
    return len(issues) == 0

# Print configuration status on import
if __name__ == "__main__":
    print("🔧 Configuration Status:")
    print(f"   - Database: {DB_CONFIG['type']}")
    print(f"   - LLM Provider: {'OpenAI' if USE_OPENAI else 'Hugging Face'}")
    print(f"   - Model: {ACTIVE_LLM_CONFIG['model']}")
    print(f"   - HF Token: {'✅ Found' if get_hf_token() else '❌ Missing'}")
    print(f"   - OpenAI Key: {'✅ Found' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
    
    is_valid = validate_config()
    print(f"   - Config Valid: {'✅ Yes' if is_valid else '❌ No'}")