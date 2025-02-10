LANGUAGE_CODE_MAP = {
    # Language to Code mapping
    'english': 'en',
    'hindi': 'hi',
    'tamil': 'ta',
    'telugu': 'te',
    'kannada': 'kn',
    'malayalam': 'ml',
    'bengali': 'bn',
    'gujarati': 'gu',
    'marathi': 'mr',
    'punjabi': 'pa',
    'odia': 'or'
}

def get_language_code(lang_name):
    return LANGUAGE_CODE_MAP.get(lang_name.lower(), 'en')