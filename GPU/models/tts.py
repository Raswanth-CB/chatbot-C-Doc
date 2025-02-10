# multilingual_chatbot/models/tts.py
import torch
from transformers import VitsModel, AutoTokenizer

class IndicTTS:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

    def synthesize(self, text, language):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model(**inputs).waveform
        
        return output.cpu().numpy().squeeze()