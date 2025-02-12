from langdetect import detect
from typing import Optional

SUPPORTED_LANGUAGES = {
    # ISO 639-1 codes for 22 scheduled Indian languages + English
    'en': 'English',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'te': 'Telugu',
    'ta': 'Tamil',
    'mr': 'Marathi',
    'ur': 'Urdu',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'pa': 'Punjabi',
    'or': 'Odia',
    'as': 'Assamese',
    'mai': 'Maithili',
    'sat': 'Santali',
    'ks': 'Kashmiri',
    'sd': 'Sindhi',
    'do': 'Dogri',
    'brx': 'Bodo',
    'mni': 'Manipuri',
    'sa': 'Sanskrit',
    'ne': 'Nepali',
    'kok': 'Konkani'
}


def detect_language(text: str) -> Optional[str]:
    """
    Detect language of input text.
    
    Args:
        text: Input text
        
    Returns:
        ISO 639-1 language code if detected language is supported,
        None otherwise
    """
    try:
        detected_lang = detect(text)
        return detected_lang if detected_lang in SUPPORTED_LANGUAGES else None
    except:
        return None
