from langdetect import detect

def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except:
        return 'en'