import json
from typing import Dict, Any


def load_deepspeed_config(config_path: str) -> Dict[str, Any]:
    """Load DeepSpeed configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_language_code(language: str) -> str:
    """
    Convert language name to ISO code.
    
    Args:
        language: Language name or code
        
    Returns:
        ISO 639-1 language code
    """
    # Mapping of language names to ISO codes
    LANGUAGE_MAP = {
        'english': 'en',
        'hindi': 'hi',
        'bengali': 'bn',
        'telugu': 'te',
        'tamil': 'ta',
        'marathi': 'mr',
        'urdu': 'ur',
        'gujarati': 'gu',
        'kannada': 'kn',
        'malayalam': 'ml',
        'punjabi': 'pa',
        'odia': 'or',
        'assamese': 'as',
        'maithili': 'mai',
        'santali': 'sat',
        'kashmiri': 'ks',
        'sindhi': 'sd',
        'dogri': 'do',
        'bodo': 'brx',
        'manipuri': 'mni',
        'sanskrit': 'sa',
        'nepali': 'ne',
        'konkani': 'kok'
    }

    lang = language.lower()
    if lang in LANGUAGE_MAP:
        return LANGUAGE_MAP[lang]
    # If input is already a valid code, return it
    if lang in LANGUAGE_MAP.values():
        return lang
    raise ValueError(f"Unsupported language: {language}")
