# multilingual_chatbot/models/asr.py
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class WhisperASR:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small"
        ).to(self.device)
        self.model.config.forced_decoder_ids = None

    def transcribe(self, audio_path, source_lang=None):
        # Load audio (replace with actual audio loading)
        inputs = self.processor(
            audio_path, 
            return_tensors="pt", 
            sampling_rate=16000
        ).input_features.to(self.device)
        
        forced_decoder_ids = None
        if source_lang:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=source_lang,
                task="transcribe"
            )
            
        generated_ids = self.model.generate(
            inputs,
            forced_decoder_ids=forced_decoder_ids
        )
        
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]