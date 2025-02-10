from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class IndicTranslator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.tokenizers = {}
        
    def _load_model(self, direction):
        if direction not in self.models:
            model_name = f"ai4bharat/indictrans2-{direction}-1B"
            self.models[direction] = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, torch_dtype=torch.float16
            ).to(self.device)
            self.tokenizers[direction] = AutoTokenizer.from_pretrained(model_name)
        
    def translate(self, text, source_lang, target_lang):
        direction = f"{source_lang}-{target_lang}"
        self._load_model(direction)
        
        inputs = self.tokenizers[direction](
            text, 
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        outputs = self.models[direction].generate(**inputs)
        return self.tokenizers[direction].decode(
            outputs[0], 
            skip_special_tokens=True
        )