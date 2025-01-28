import argparse
from utils.ds_wrappers import load_model
from utils.language_utils import detect_language, create_translator, translate
from utils.audio_utils import load_audio_model, transcribe_audio, create_tts_engine
import torch

def main_pipeline(args):
    # Initialize models with DeepSpeed
    translator = create_translator()
    asr_pipeline = load_audio_model()
    tts_engine = create_tts_engine()
    llama_model = load_model("llama")

    # Process input
    if args.input_type == "audio":
        input_text = transcribe_audio(args.input, asr_pipeline)
    else:
        with open(args.input, "r") as f:
            input_text = f.read()

    # Detect and translate to English
    src_lang = detect_language(input_text)
    en_text = translate(input_text, src_lang, "en", translator)

    # Process with Llama
    inputs = llama_model.tokenizer(en_text, return_tensors="pt").to(llama_model.device)
    outputs = llama_model.generate(**inputs)
    processed_text = llama_model.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Generate output
    if args.output_format == "audio":
        audio = text_to_speech(processed_text, args.output_lang, tts_engine)
        torchaudio.save(args.output, audio["audio"], sample_rate=audio["sampling_rate"])
    else:
        translated_output = translate(processed_text, "en", args.output_lang, translator)
        with open(args.output, "w") as f:
            f.write(translated_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Translation Pipeline")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--input-type", choices=["text", "audio"], required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--output-lang", type=str, default="hi")
    parser.add_argument("--output-format", choices=["text", "audio"], default="text")
    
    args = parser.parse_args()
    main_pipeline(args)
