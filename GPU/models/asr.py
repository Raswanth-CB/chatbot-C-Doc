from transformers import pipeline
import torch

class WhisperASR:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            device=self.device,
            torch_dtype=torch.float16
        )
    
    def transcribe(self, audio_path, source_lang=None):
        result = self.pipe(
            audio_path,
            generate_kwargs={"language": source_lang} if source_lang else None,
            return_tensors="pt"
        )
        return result["text"], result["language"]