import whisper
import torch
from typing import Optional, Union
import numpy as np


class WhisperASR:
    def __init__(self, model_name: str = "large-v3"):
        """Initialize Whisper ASR model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_name).to(self.device)

    def transcribe(
        self,
        audio_path: Union[str, np.ndarray],
        source_lang: Optional[str] = None
    ) -> dict:
        """
        Transcribe audio to text.
        
        Args:
            audio_path: Path to audio file or numpy array of audio data
            source_lang: Source language code (optional)
            
        Returns:
            Dictionary containing transcription and metadata
        """
        transcription_options = {}
        if source_lang:
            transcription_options["language"] = source_lang

        result = self.model.transcribe(
            audio_path,
            **transcription_options
        )

        return {
            "text": result["text"],
            "language": result["language"],
            "segments": result["segments"]
        }

    def __call__(self, *args, **kwargs):
        return self.transcribe(*args, **kwargs)
