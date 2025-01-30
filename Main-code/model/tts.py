from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile as wavfile
from typing import Optional, Union
import numpy as np


class IndicTTS:
    def __init__(self, model_name: str = "ai4bharat/indic-parler-tts-pretrained"):
        """Initialize Indic-Parler TTS model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = VitsModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def synthesize(
        self,
        text: str,
        language: str,
        output_path: Optional[str] = None,
        speaker_id: int = 0
    ) -> Union[np.ndarray, None]:
        """
        Convert text to speech.
        
        Args:
            text: Input text
            language: Language code
            output_path: Path to save audio file (optional)
            speaker_id: Speaker identity for multi-speaker models
            
        Returns:
            Numpy array of audio data if output_path is None,
            otherwise saves to file and returns None
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate_speech(
                inputs["input_ids"],
                speaker_ids=torch.tensor([speaker_id]).to(self.device),
                language_ids=torch.tensor(
                    [self.tokenizer.lang2id[language]]).to(self.device)
            )

        # Convert to numpy array
        audio_data = output.cpu().numpy()

        if output_path:
            wavfile.write(
                output_path, self.model.config.sampling_rate, audio_data)
            return None

        return audio_data

    def __call__(self, *args, **kwargs):
        return self.synthesize(*args, **kwargs)
