from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from typing import Optional


class IndicTranslator:
    def __init__(self, model_name: str = "ai4bharat/indictrans2-indic-indic-1B"):
        """Initialize IndicTrans2 model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """
        Translate text between Indian languages.
        
        Args:
            text: Input text
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        # Prepare input with language tags
        input_text = f">>{source_lang}<< {text} >>{target_lang}<<"

        # Tokenize and generate
        inputs = self.tokenizer(
            input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=5,
                length_penalty=1.0,
                early_stopping=True
            )

        translated_text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True)
        return translated_text

    def __call__(self, *args, **kwargs):
        return self.translate(*args, **kwargs)
