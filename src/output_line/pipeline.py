from inference_ttt import translate_text
from inference_tts import generate_audio

def main():
    print("Welcome to the Text-to-Speech pipeline!")
    input_text = input("Enter the English text to translate and convert to speech: ").strip()

    if not input_text:
        print("Error: No input provided.")
        return

    # Step 1: Text-to-Text Translation
    print("\nTranslating text to Indian languages...")
    translations = translate_text(input_text)
    print("Translations complete. Saved to 'translations/'.")

    # Step 2: Text-to-Speech Conversion
    print("\nConverting translations to speech...")
    generate_audio()
    print("Audio generation complete. Saved to 'audio/'.")

    print("\nPipeline finished successfully!")

if __name__ == "__main__":
    main()
