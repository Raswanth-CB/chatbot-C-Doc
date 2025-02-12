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
        # GPU setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize models
        self.asr_model = WhisperASR()
        self.translator = IndicTranslator()
        self.tts_model = IndicTTS()
        self.llama_model = LlamaModel()

        # Conversation history
        self.conversation_history = []

    def process_input(self, input_data, input_type="text", input_language=None, output_type="text", output_language=None):
        try:
            # Audio input processing
            if input_type == "audio":
                print("Processing audio input...")
                text = self.asr_model.transcribe(input_data, source_lang=input_language)
            else:
                text = input_data

            # Language detection
            src_lang = input_language or detect_language(text)
            src_lang = get_language_code(src_lang)
            print(f"Detected language: {src_lang}")
            output_lang = output_language or src_lang

            # Translation to English
            if src_lang != "en":
                english_text = self.translator.translate(text, src_lang, "en")
            else:
                english_text = text

            # Generate response
            prompt = self._prepare_prompt(english_text)
            english_response = self.llama_model.generate_response(prompt)

            # Reverse translation
            if output_lang != "en":
                final_text = self.translator.translate(english_response, "en", output_lang)
            else:
                final_text = english_response

            self.conversation_history.append({"user": text, "assistant": final_text})

            # Generate output
            if output_type == "audio":
                audio = self.tts_model.synthesize(final_text, output_lang)
                return audio
            elif output_type == "both":
                audio = self.tts_model.synthesize(final_text, output_lang)
                return final_text, audio
            return final_text

        except Exception as e:
            return f"Error processing request: {str(e)}"

    def _prepare_prompt(self, english_text):
        prompt = """You are a helpful multilingual assistant. Keep responses under 2 sentences.\n\n"""
        for turn in self.conversation_history[-3:]:  # Keep last 3 exchanges
            prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        prompt += f"User: {english_text}\nAssistant:"
        return prompt

def main():
    chatbot = MultilingualChatbot()
    
    # Example usage
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        response = chatbot.process_input(
            input_data=user_input,
            input_type="text",
            output_type="text"
        )
        print(f"\nAssistant: {response}")

if __name__ == "__main__":
    main()