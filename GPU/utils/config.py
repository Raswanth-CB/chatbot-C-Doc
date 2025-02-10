# multilingual_chatbot/utils/config.py

LANGUAGE_CODES = {
    "english": "en",
    "hindi": "hi",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "bengali": "bn",
    "malayalam": "ml",
    "gujarati": "gu",
    "marathi": "mr"
}

def get_language_code(lang):
    return LANGUAGE_CODES.get(lang.lower(), "en")