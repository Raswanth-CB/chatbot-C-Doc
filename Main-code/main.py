import os
from typing import Optional, Union, Tuple, Dict, Any
import numpy as np
from models.asr import WhisperASR
from models.translation import IndicTranslator
from models.tts import IndicTTS
from models.llama import LlamaModel
from utils.language_detect import detect_language
from utils.config import load_deepspeed_config, get_language_code


class MultilingualChatbot:
    def __init__(self, deepspeed_config_path: str):
        """Initialize multilingual chatbot with all components."""
        # Load DeepSpeed configuration
        self.deepspeed_config = load_deepspeed_config(deepspeed_config_path)

        # Initialize models
        self.asr_model = WhisperASR()
        self.translator = IndicTranslator()
        self.tts_model = IndicTTS()
        self.llama_model = LlamaModel(deepspeed_config=self.deepspeed_config)

        # Store conversation history
        self.conversation_history = []

    def detect_and_validate_language(
        self,
        text: str,
        provided_language: Optional[str] = None
    ) -> str:
        """
        Detect language and validate against supported languages.
        
        Args:
            text: Input text
            provided_language: Language code if provided
            
        Returns:
            Validated language code
        """
        if provided_language:
            lang_code = get_language_code(provided_language)
        else:
            detected_lang = detect_language(text)
            if not detected_lang:
                raise ValueError(
                    "Could not detect language. Please specify input language.")
            lang_code = detected_lang

        return lang_code

    def prepare_llama_prompt(
        self,
        text: str,
        language: str
    ) -> str:
        """
        Prepare prompt for LLaMA model with conversation history.
        
        Args:
            text: Current input text
            language: Language code
            
        Returns:
            Formatted prompt string
        """
        prompt = "You are a helpful assistant that can converse in multiple Indian languages.\n\n"

        # Add conversation history
        # Keep last 5 turns for context
        for turn in self.conversation_history[-5:]:
            prompt += f"User: {turn['user']
                               }\nAssistant: {turn['assistant']}\n\n"

        # Add current query
        prompt += f"User: {text}\nAssistant:"

        return prompt

    def process_input(
        self,
        input_data: Union[str, np.ndarray],
        input_type: str,
        input_language: Optional[str] = None,
        output_type: str = "text",
        output_language: Optional[str] = None
    ) -> Union[str, np.ndarray, Tuple[str, np.ndarray]]:
        """
        Process input through the complete pipeline.
        
        Args:
            input_data: Input text or audio data
            input_type: Type of input ("text" or "audio")
            input_language: Input language code (optional)
            output_type: Desired output type ("text" or "audio")
            output_language: Desired output language code (optional)
            
        Returns:
            Processed output as text, audio data, or both
        """
        # Step 1: Convert input to text if needed
        if input_type == "audio":
            text = self.asr_model.transcribe(
                input_data, source_lang=input_language)["text"]
        else:
            text = input_data

        # Step 2: Detect and validate language
        input_lang = self.detect_and_validate_language(text, input_language)
        output_lang = output_language or input_lang

        # Step 3: Translate to English if needed
        if input_lang != "en":
            english_text = self.translator.translate(text, input_lang, "en")
        else:
            english_text = text

        # Step 4: Generate response using LLaMA
        prompt = self.prepare_llama_prompt(english_text, input_lang)
        english_response = self.llama_model.generate_response(prompt)

        # Step 5: Translate response if needed
        if output_lang != "en":
            final_text = self.translator.translate(
                english_response, "en", output_lang)
        else:
            final_text = english_response

        # Store conversation turn
        self.conversation_history.append({
            "user": text,
            "assistant": final_text
        })

        # Step 6: Convert to audio if requested
        if output_type == "audio":
            audio_data = self.tts_model.synthesize(final_text, output_lang)
            if output_type == "both":
                return final_text, audio_data
            return audio_data

        return final_text

    def save_conversation(self, filepath: str):
        """Save conversation history to file."""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f,
                      ensure_ascii=False, indent=2)

    def load_conversation(self, filepath: str):
        """Load conversation history from file."""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            self.conversation_history = json.load(f)


def main():
    """Example usage of the MultilingualChatbot."""
    # Initialize chatbot
    config_path = "deepspeed_config.json"
    chatbot = MultilingualChatbot(config_path)

    # Example: Text input in Hindi, text output in English
    hindi_text = "नमस्ते, कैसे हो आप?"
    response = chatbot.process_input(
        input_data=hindi_text,
        input_type="text",
        input_language="hi",
        output_type="text",
        output_language="en"
    )
    print(f"Hindi Input: {hindi_text}")
    print(f"English Response: {response}")

    # Example: Audio input in Tamil, audio output in Tamil
    tamil_audio_path = "input_audio.wav"
    if os.path.exists(tamil_audio_path):
        response = chatbot.process_input(
            input_data=tamil_audio_path,
            input_type="audio",
            input_language="ta",
            output_type="both",  # Get both text and audio
            output_language="ta"
        )

        if isinstance(response, tuple):
            text_response, audio_data = response
            print(f"Tamil Response Text: {text_response}")
            # Save audio response
            from scipy.io.wavfile import write
            # Assuming 24kHz sample rate
            write("output_audio.wav", 24000, audio_data)


if __name__ == "__main__":
    main()
