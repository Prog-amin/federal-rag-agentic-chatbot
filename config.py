import os
import aiohttp
import json
from dotenv import load_dotenv

load_dotenv()

# Database Configuration (SQLite for HF Spaces)
DB_CONFIG = {
    'type': 'sqlite',
    'db_path': 'federal_registry.db'
}

def get_llm_config():
    """Get LLM configuration based on provider"""
    provider = os.getenv('LLM_PROVIDER', 'huggingface').lower()
    
    configs = {
        'openai_compatible': {
            'provider': 'openai_compatible',
            'base_url': os.getenv('LLM_BASE_URL', 'https://api.sambanova.ai/v1'),
            'model': os.getenv('LLM_MODEL', 'Meta-Llama-3.1-8B-Instruct'),
            'api_key': os.getenv('LLM_API_KEY'),
            'use_openai_format': True
        },
        'groq': {
            'provider': 'groq',
            'base_url': 'https://api.groq.com/openai/v1',
            'model': os.getenv('LLM_MODEL', 'llama-3.1-8b-instant'),
            'api_key': os.getenv('LLM_API_KEY'),
            'use_openai_format': True
        },
        'deepseek': {
            'provider': 'deepseek',
            'base_url': 'https://api.deepseek.com/v1',
            'model': os.getenv('LLM_MODEL', 'deepseek-chat'),
            'api_key': os.getenv('LLM_API_KEY'),
            'use_openai_format': True
        },
        'huggingface': {
            'provider': 'huggingface',
            'base_url': 'https://api-inference.huggingface.co/models',
            'model': os.getenv('LLM_MODEL', 'microsoft/DialoGPT-medium'),
            'api_key': os.getenv('HUGGINGFACE_API_TOKEN') or os.getenv('HF_TOKEN'),
            'use_openai_format': False
        }
    }
    
    return configs.get(provider, configs['huggingface'])

# Use the dynamic LLM configuration
LLM_CONFIG = get_llm_config()

# Federal Registry API Configuration
FEDERAL_REGISTRY_API = {
    'base_url': 'https://www.federalregister.gov/api/v1',
    'documents_endpoint': '/documents.json',
    'rate_limit_delay': 1.0
}

# Data Pipeline Configuration
PIPELINE_CONFIG = {
    'data_dir': '/tmp/data',
    'raw_data_dir': '/tmp/data/raw',
    'processed_data_dir': '/tmp/data/processed',
    'retention_days': 7,
    'max_pages_per_run': 10,
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
    'max_memory_mb': 16 * 1024,
    'max_cpu_cores': 8,
    'timeout_seconds': 120,
    'enable_persistence': True,
    'temp_dir': '/tmp'
}

def get_hf_token():
    """Get Hugging Face token from environment"""
    token = os.getenv('HUGGINGFACE_API_TOKEN') or os.getenv('HF_TOKEN')
    if not token:
        print("⚠️  Warning: No Hugging Face token found.")
    return token

def validate_config():
    """Validate configuration settings"""
    issues = []
    
    if LLM_CONFIG['provider'] != 'huggingface' and not LLM_CONFIG.get('api_key'):
        issues.append(f"API key not configured for {LLM_CONFIG['provider']}")
    
    if LLM_CONFIG['provider'] == 'huggingface' and not get_hf_token():
        issues.append("Hugging Face token not configured")
    
    if issues:
        print("⚠️  Configuration issues:")
        for issue in issues:
            print(f"   - {issue}")
    
    return len(issues) == 0

# Simplified LLM API call function
async def call_llm(prompt, max_tokens=512, temperature=0.7):
    """Make API call to configured LLM provider"""
    try:
        if LLM_CONFIG['use_openai_format']:
            # OpenAI-compatible APIs (SambaNova, Groq, DeepSeek)
            url = f"{LLM_CONFIG['base_url']}/chat/completions"
            payload = {
                "model": LLM_CONFIG['model'],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            headers = {
                "Authorization": f"Bearer {LLM_CONFIG['api_key']}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    else:
                        error_text = await response.text()
                        print(f"API Error {response.status}: {error_text}")
                        return None
        
        else:
            # Hugging Face Inference API
            url = f"{LLM_CONFIG['base_url']}/{LLM_CONFIG['model']}"
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "return_full_text": False
                }
            }
            headers = {
                "Authorization": f"Bearer {LLM_CONFIG['api_key']}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, list) and len(result) > 0:
                            return result[0].get('generated_text', '')
                        return str(result)
                    else:
                        error_text = await response.text()
                        print(f"HF API Error {response.status}: {error_text}")
                        return None
                        
    except aiohttp.ClientTimeout:
        print("API request timed out")
        return None
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None

# Test function to verify LLM is working
async def test_llm():
    """Test the LLM configuration"""
    print(f"Testing {LLM_CONFIG['provider']} with model {LLM_CONFIG['model']}...")
    response = await call_llm("Hello! Please respond with 'API working correctly.'")
    if response:
        print(f"✅ LLM Test Success: {response}")
        return True
    else:
        print("❌ LLM Test Failed")
        return False

if __name__ == "__main__":
    print("🔧 Configuration Status:")
    print(f"   - Database: {DB_CONFIG['type']}")
    print(f"   - LLM Provider: {LLM_CONFIG['provider']}")
    print(f"   - LLM Model: {LLM_CONFIG['model']}")
    print(f"   - LLM Base URL: {LLM_CONFIG['base_url']}")
    
    is_valid = validate_config()
    print(f"   - Config Valid: {'✅ Yes' if is_valid else '❌ No'}")
    
    # Test the LLM if config is valid
    if is_valid:
        import asyncio
        asyncio.run(test_llm())