# multilingual_chatbot/main.py

import os
import torch
from models.asr import WhisperASR
from models.translation import IndicTranslator
from models.tts import IndicTTS
from models.llama import LlamaModel
from utils.language_detect import detect_language
from utils.config import get_language_code

class MultilingualChatbot:
    def __init__(self):
        # Initialize models with GPU support
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.asr_model = WhisperASR().to(self.device)
        self.translator = IndicTranslator()
        self.tts_model = IndicTTS().to(self.device)
        self.llama_model = LlamaModel().to(self.device)

        # Conversation history
        self.conversation_history = []

    def process_input(
        self,
        input_data,
        input_type="text",
        input_language=None,
        output_type="text",
        output_language=None
    ):
        if input_type == "audio":
            print("Processing audio input...")
            text = self.asr_model.transcribe(input_data, source_lang=input_language)
        else:
            print("Processing text input...")
            text = input_data

        # Language detection
        src_lang = input_language or detect_language(text)
        src_lang = get_language_code(src_lang)
        print(f"Source language: {src_lang}")
        output_lang = output_language or src_lang

        # Translation to English
        if src_lang != "en":
            english_text = self.translator.translate(text, src_lang, "en")
        else:
            english_text = text

        # LLM Response
        prompt = self._prepare_prompt(english_text)
        english_response = self.llama_model.generate_response(prompt)

        # Translate back if needed
        if output_lang != "en":
            final_text = self.translator.translate(english_response, "en", output_lang)
        else:
            final_text = english_response

        self.conversation_history.append({
            "user": text,
            "assistant": final_text
        })

        # Generate output
        if output_type == "audio":
            audio_data = self.tts_model.synthesize(final_text, output_lang)
            return audio_data
        elif output_type == "both":
            audio_data = self.tts_model.synthesize(final_text, output_lang)
            return final_text, audio_data
        return final_text

    def _prepare_prompt(self, english_text):
        prompt = "You are a helpful multilingual assistant. Keep responses concise.\n\n"
        for turn in self.conversation_history[-3:]:
            prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        prompt += f"User: {english_text}\nAssistant:"
        return prompt

def main():
    chatbot = MultilingualChatbot()
    
    # Example text input
    user_text = input("Enter your message: ")
    response = chatbot.process_input(
        input_data=user_text,
        input_type="text",
        output_type="text"
    )
    print("\nResponse:", response)

if __name__ == "__main__":
    main()