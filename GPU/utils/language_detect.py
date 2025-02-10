import fasttext
from utils.config import LANGUAGE_CODES

class LanguageDetector:
    def __init__(self):
        self.model = fasttext.load_model("lid.176.ftz")
    
    def detect(self, text):
        predictions = self.model.predict(text, k=1)
        lang_code = predictions[0][0].split("__")[-1]
        return LANGUAGE_CODES.get(lang_code, "en")