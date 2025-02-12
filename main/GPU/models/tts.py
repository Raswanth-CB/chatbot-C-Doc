import torch
from transformers import AutoModel, AutoTokenizer

class IndicTTS:
    def __init__(self):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-enabled GPU is required but not available")
        self.model = AutoModel.from_pretrained("model/indic-parler-tts-pretrained").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("model/indic-parler-tts-pretrained")
        
    def synthesize(self, text, language):
        # Set language code based on config mapping (ensure language codes match the model's requirements)
        inputs = self.tokenizer(
            text=text,
            language=language,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            output = self.model(**inputs).waveform
        return output.cpu().numpy().squeeze()