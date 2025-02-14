# multilingual_chatbot/models/asr.py
import torch
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class WhisperASR:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = WhisperProcessor.from_pretrained("/home/miphi/raswanth/model/whisper-large-v3")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "/home/miphi/raswanth/model/whisper-large-v3"
        ).to(self.device)
        self.model.config.forced_decoder_ids = None

    def transcribe(self, audio_path, source_lang=None):
        # Load and preprocess audio
        audio_input, sample_rate = sf.read(audio_path)
        if sample_rate != 16000:
            audio_input = self._resample_audio(audio_input, sample_rate)
            
        inputs = self.processor(
            audio_input,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device)
        
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=source_lang, task="transcribe"
        ) if source_lang else None
            
        generated_ids = self.model.generate(inputs, forced_decoder_ids=forced_decoder_ids)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _resample_audio(self, audio, orig_sr):
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=16000)