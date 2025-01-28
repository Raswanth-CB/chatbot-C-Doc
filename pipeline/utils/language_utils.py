from langdetect import detect, DetectorFactory
from transformers import pipeline
import logging

DetectorFactory.seed = 0

def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        logging.warning(f"Language detection failed: {e}")
        return "en"

def create_translator():
    return pipeline(
        "translation",
        model="ai4bharat/indictrans2-indic-indic-1B",
        device_map="auto"
    )

def translate(text, src_lang, tgt_lang, translator):
    if src_lang == tgt_lang:
        return text
    return translator(
        text,
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )[0]["translation_text"]
