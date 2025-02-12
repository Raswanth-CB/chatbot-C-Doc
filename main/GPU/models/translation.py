import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class IndicTranslator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-1B").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-1B")

    def translate(self, text, src_lang, tgt_lang):
        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        generated_tokens = self.model.generate(**inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang])
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]