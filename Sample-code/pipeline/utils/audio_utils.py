from transformers import pipeline
import torchaudio

def load_audio_model():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        device_map="auto"
    )

def transcribe_audio(audio_path, asr_pipeline):
    return asr_pipeline(audio_path)["text"]

def create_tts_engine():
    return pipeline(
        "text-to-speech",
        model="ai4bharat/indic-parler-tts-pretrained",
        device_map="auto"
    )

def text_to_speech(text, language, tts_pipeline):
    return tts_pipeline(text, lang=language)
