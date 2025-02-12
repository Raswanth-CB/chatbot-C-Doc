import torch
import deepspeed
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech
)
from langdetect import detect
import soundfile as sf

# DeepSpeed configuration with NVMe offloading
ds_config = {
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_param": {"device": "nvme", "nvme_path": "./nvme"},
        "offload_optimizer": {"device": "nvme", "nvme_path": "./nvme"},
        "overlap_comm": True,
        "contiguous_gradients": True
    },
    "trainer": {"use_cpu_initialization": True},
    "steps_per_print": 1,
    "train_micro_batch_size_per_gpu": 1
}


class MultilingualChatbot:
    def __init__(self):
        # Initialize components with DeepSpeed
        self.asr_processor = None
        self.asr_model = None
        self.translation_models = {}
        self.tts_models = {}
        self.llama_model = None
        self.llama_tokenizer = None

        self.load_models()

    def load_models(self):
        """Load all required models with DeepSpeed optimization"""
        # Audio-to-Text (Whisper)
        self.asr_processor = WhisperProcessor.from_pretrained(
            "openai/whisper-large-v3")
        asr_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3")
        self.asr_model = deepspeed.init_inference(
            asr_model,
            config=ds_config,
            dtype=torch.float16,
            replace_with_kernel_inject=True
        )

        # Text-to-Text Translation (Indictrans2)
        translation_model = AutoModelForSeq2SeqLM.from_pretrained(
            "ai4bharat/indictrans2-indic-indic-1B")
        self.translation_model = deepspeed.init_inference(
            translation_model,
            config=ds_config,
            dtype=torch.float16,
            replace_with_kernel_inject=True
        )
        self.trans_tokenizer = AutoTokenizer.from_pretrained(
            "ai4bharat/indictrans2-indic-indic-1B")

        # Main LLM (Llama-3.1-8B)
        self.llama_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct")
        llama_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct")
        self.llama_model = deepspeed.init_inference(
            llama_model,
            config=ds_config,
            dtype=torch.float16,
            replace_with_kernel_inject=True
        )

        # Text-to-Speech (Indic-Parler-TTS)
        tts_model = SpeechT5ForTextToSpeech.from_pretrained(
            "ai4bharat/indic-parler-tts-pretrained")
        self.tts_model = deepspeed.init_inference(
            tts_model,
            config=ds_config,
            dtype=torch.float16,
            replace_with_kernel_inject=True
        )
        self.tts_processor = SpeechT5Processor.from_pretrained(
            "ai4bharat/indic-parler-tts-pretrained")

    def detect_language(self, text):
        """Detect input language with fallback to English"""
        try:
            lang = detect(text)
            return lang if lang in SUPPORTED_LANGS else 'en'
        except:
            return 'en'

    def translate_text(self, text, src_lang, tgt_lang):
        """Translate text between supported languages"""
        self.trans_tokenizer.src_lang = src_lang
        inputs = self.trans_tokenizer(text, return_tensors="pt").to("cuda")
        translated = self.translation_model.generate(
            **inputs, forced_bos_token_id=self.trans_tokenizer.lang_code_to_id[tgt_lang]
        )
        return self.trans_tokenizer.decode(translated[0], skip_special_tokens=True)

    def process_audio_input(self, audio_path):
        """Convert audio to text"""
        audio_input = self.asr_processor(
            audio_path, sampling_rate=16000, return_tensors="pt"
        ).input_features.to("cuda")
        predicted_ids = self.asr_model.generate(audio_input)
        return self.asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    def generate_response(self, prompt):
        """Generate response using LLaMA"""
        inputs = self.llama_tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.llama_model.generate(**inputs, max_length=512)
        return self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def text_to_speech(self, text, lang):
        """Convert text to speech in target language"""
        inputs = self.tts_processor(text=text, return_tensors="pt").to("cuda")
        speech = self.tts_model.generate_speech(inputs["input_ids"])
        return speech.cpu().numpy()

    def process_pipeline(self, input_data, input_type="text", output_type="text"):
        """Main processing pipeline"""
        # Process input
        if input_type == "audio":
            raw_text = self.process_audio_input(input_data)
        else:
            raw_text = input_data

        # Detect language
        src_lang = self.detect_language(raw_text)

        # Translate to English if needed
        if src_lang != 'en':
            english_text = self.translate_text(raw_text, src_lang, 'en')
        else:
            english_text = raw_text

        # Generate response
        english_response = self.generate_response(english_text)

        # Translate back to source language
        if src_lang != 'en':
            final_response = self.translate_text(
                english_response, 'en', src_lang)
        else:
            final_response = english_response

        # Generate output
        output_audio = None
        if output_type == "audio":
            output_audio = self.text_to_speech(final_response, src_lang)

        return {
            "text": final_response,
            "audio": output_audio,
            "language": src_lang
        }


# Supported languages (example list - verify with model documentation)
SUPPORTED_LANGS = ['en', 'hi', 'ta', 'te', 'kn',
                   'ml', 'mr', 'bn', 'gu', 'pa', 'or', 'as', 'ur']

# Usage example
if __name__ == "__main__":
    chatbot = MultilingualChatbot()

    # Process Tamil audio input
    result = chatbot.process_pipeline(
        input_data="tamil_audio.wav",
        input_type="audio",
        output_type="audio"
    )

    # Save audio output
    if result['audio'] is not None:
        sf.write('output.wav', result['audio'], 16000)

    print("Text Response:", result['text'])
