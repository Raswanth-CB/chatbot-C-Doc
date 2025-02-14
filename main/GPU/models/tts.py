import os
import torch
import logging
import soundfile as sf
from datetime import datetime
from models.asr import WhisperASR
from models.translation import IndicTranslator
from models.tts import IndicTTS
from models.llama import LlamaModel
from utils.language_detect import detect_language
from utils.config import get_language_code


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("/home/miphi/raswanth/chatbot-C-Doc/main/GPU/log", 
                                       f'pipeline_log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class MultilingualChatbot:
    def __init__(self):
        # Force GPU usage and read from environment variable
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-enabled GPU is required but not available")
        
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        logger.info(f"Initializing models on CUDA devices: {visible_devices or 'all available'}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using primary device: {self.device}")

        # Initialize models with automatic device placement
        # self.asr_model = WhisperASR().to(self.device)
        # self.translator = IndicTranslator().to(self.device)
        # self.tts_model = IndicTTS().to(self.device)
        # self.llama_model = LlamaModel().to(self.device)
        
        # Initialize models (REMOVE .to() calls)
        self.asr_model = WhisperASR()  # Fixed
        self.translator = IndicTranslator()  # Fixed
        self.tts_model = IndicTTS()  # Fixed
        self.llama_model = LlamaModel()  # Fixed


        self.conversation_history = []
        logger.info("All models initialized successfully")

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

        # Language handling
        src_lang = input_language or detect_language(text)
        src_lang = get_language_code(src_lang)
        output_lang = output_language or src_lang

        # Translation to English
        if src_lang != "en":
            english_text = self.translator.translate(text, src_lang, "en")
        else:
            english_text = text

        # Generate response
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
        timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
        output_dir = "/home/miphi/raswanth/chatbot-C-Doc/main/GPU/log/output"
        log_dir = "/home/miphi/raswanth/chatbot-C-Doc/main/GPU/log"
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        if output_type == "audio":
            audio_filename = os.path.join(output_dir, f"output_{timestamp}.wav")
            audio = self.tts_model.synthesize(final_text, output_lang)
            sf.write(audio_filename, audio, 16000)
            return f"Audio response saved to {audio_filename}"
        elif output_type == "both":
            audio_filename = os.path.join(output_dir, f"output_{timestamp}.wav")
            audio = self.tts_model.synthesize(final_text, output_lang)
            sf.write(audio_filename, audio, 16000)
            return final_text, audio_filename
        return final_text
    
        # # Generate output
        # if output_type == "audio":
        #     audio = self.tts_model.synthesize(final_text, output_lang)
        #     sf.write("output.wav", audio, 16000)
        #     return "Audio response saved to output.wav"
        # elif output_type == "both":
        #     audio = self.tts_model.synthesize(final_text, output_lang)
        #     sf.write("output.wav", audio, 16000)
        #     return final_text, "output.wav"
        # return final_text

    def _prepare_prompt(self, english_text):
        prompt = "You are a helpful multilingual assistant. Keep responses concise.\n\n"
        for turn in self.conversation_history[-3:]:
            prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        prompt += f"User: {english_text}\nAssistant:"
        return prompt

def main():
    try:
        chatbot = MultilingualChatbot()  # No device IDs in constructor
        logger.info("Chatbot initialized successfully")
        
        input_type = input("Input type (text/audio): ")
        input_data = input("Enter text or audio path: ")
        output_type = input("Output type (text/audio/both): ")
        
        response = chatbot.process_input(
            input_data=input_data,
            input_type=input_type,
            output_type=output_type
        )
        
        if isinstance(response, tuple):
            print("\nText Response:", response[0])
            print("Audio saved to:", response[1])
        else:
            print("\nResponse:", response)

    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()