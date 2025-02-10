from transformers import pipeline
import torch

class IndicTTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline(
            "text-to-speech",
            model="ai4bharat/indic-parler-tts-pretrained",
            device=self.device,
            torch_dtype=torch.float16
        )
    
    def synthesize(self, text, language):
        output = self.pipe(
            text,
            target_language=language,
            target_voice="female"
        )
        return output["audio"]