import torch
from transformers import VitsModel, AutoTokenizer

class IndicTTS:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {
            'en': (VitsModel.from_pretrained("facebook/mms-tts-eng"), AutoTokenizer.from_pretrained("facebook/mms-tts-eng")),
            'hi': (VitsModel.from_pretrained("facebook/mms-tts-hin"), AutoTokenizer.from_pretrained("facebook/mms-tts-hin")),
            'ta': (VitsModel.from_pretrained("facebook/mms-tts-tam"), AutoTokenizer.from_pretrained("facebook/mms-tts-tam"))
        }
        
    def synthesize(self, text, language):
        if language not in self.models:
            language = 'en'
            
        model, tokenizer = self.models[language]
        model = model.to(self.device)
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = model(**inputs).waveform
        return output.cpu().numpy().squeeze()