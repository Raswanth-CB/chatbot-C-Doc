# Chat Bot
Complete project work flow 


## Models : 
1. text-to-text  -- ai4bharat/indictrans2-indic-indic-1B
2. text-to-speech -- ai4bharat/indic-parler-tts-pretrained
3. audio-to-text -- openai/whisper-large-v3-turbo
4. main model -- meta-llama/Llama-3.1-8B-Instruct



# Data flow 
```mermaid
    [Start]
    │
    ├── Input Handling
    │   ├── Audio Input Path → (process_audio_input)
    │   │   └→ Whisper ASR → Raw Text
    │   │
    │   └── Text Input → Directly Passed
    │
    ├── Language Detection → (detect_language)
    │   └→ Source Language (src_lang)
    │
    ├── Translation Pipeline
    │   ├── If src_lang ≠ English:
    │   │   ├→ indic-trans2 Translation (src_lang → en)
    │   │   └→ English Text
    │   │
    │   └── If src_lang = English: Direct Pass
    │
    ├── LLM Processing → (generate_response)
    │   └→ Llama-3.1-8B → English Response
    │
    ├── Reverse Translation
    │   ├── If src_lang ≠ English:
    │   │   ├→ indic-trans2 Translation (en → src_lang)
    │   │   └→ Final Translated Text
    │   │
    │   └── If src_lang = English: Direct Pass
    │
    ├── Output Generation
    │   ├── Text Output → Final Response
    │   │
    │   └── If Audio Output Required:
    │       ├→ indic-parler-tts Processing
    │       └→ Speech Audio File
    │
    [End]


```mermaid
graph TD
    A[Input] --> B{Type?}
    B --> |Audio| C[ASR: Whisper]
    B --> |Text| D[Language Detection]
    C --> D
    D --> E{English?}
    E --> |No| F[Translate to English]
    E --> |Yes| G[LLaMA Processing]
    F --> G
    G --> H{Output Language?}
    H --> |Non-English| I[Back-Translate]
    H --> |English| J[Direct Output]
    I --> K[TTS: Indic-Parler]
    J --> K
    K --> L[Output]
