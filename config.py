import os
from dotenv import load_dotenv

load_dotenv()

# Database Configuration (SQLite for HF Spaces)
DB_CONFIG = {
    'type': 'sqlite',
    'db_path': 'federal_registry.db'
}

# Local LLM Configuration (llama.cpp server)
LOCAL_LLM_CONFIG = {
    'base_url': os.getenv('LLM_BASE_URL', 'http://localhost:7861/v1'),  # Fixed port
    'model': os.getenv('LLM_MODEL', 'TinyLlama'),
    'api_key': os.getenv('LLM_API_KEY', 'not-needed'),
    'use_openai_format': True
}

# Use local LLM configuration
LLM_CONFIG = LOCAL_LLM_CONFIG

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
    
    # Check if we have valid LLM configuration
    if not LOCAL_LLM_CONFIG['base_url']:
        issues.append("LLM base URL not configured")
    
    if issues:
        print("⚠️  Configuration issues:")
        for issue in issues:
            print(f"   - {issue}")
    
    return len(issues) == 0

# Print configuration status on import
if __name__ == "__main__":
    print("🔧 Configuration Status:")
    print(f"   - Database: {DB_CONFIG['type']}")
    print(f"   - LLM Model: {LLM_CONFIG['model']}")
    print(f"   - LLM Base URL: {LLM_CONFIG['base_url']}")
    
    is_valid = validate_config()
    print(f"   - Config Valid: {'✅ Yes' if is_valid else '❌ No'}")