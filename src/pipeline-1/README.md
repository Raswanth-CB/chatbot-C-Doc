
---

# 🚀 Multilingual AI Chatbot with DeepSpeed Optimization  

This project implements a **Multilingual AI Chatbot** optimized with **DeepSpeed** for efficient inference on large models. It supports **speech-to-text (ASR), text translation, conversational AI, and text-to-speech (TTS)** functionalities.  

## ✨ **Features**  

- **Speech Recognition (ASR)**: Converts audio input to text using **OpenAI Whisper**.  
- **Language Detection**: Automatically detects the language of the input text.  
- **Text Translation**: Uses **IndicTrans2** to translate between supported languages.  
- **Conversational AI**: Uses **LLaMA-3.1-8B** for generating responses.  
- **Text-to-Speech (TTS)**: Converts responses to speech using **Indic-Parler-TTS**.  
- **DeepSpeed Acceleration**: Optimized with **NVMe offloading** for handling large models efficiently.  

## 🛠 **Installation**  

```bash
pip install torch deepspeed transformers langdetect soundfile 
```

## 📌 **Usage**  

### 1️⃣ **Run the Chatbot**  

```python
from chatbot import MultilingualChatbot  

chatbot = MultilingualChatbot()  
response = chatbot.process_pipeline(input_data="Hello, how are you?", input_type="text", output_type="text")  
print("Response:", response["text"])  
```

### 2️⃣ **Process Audio Input**  

```python
response = chatbot.process_pipeline(input_data="input_audio.wav", input_type="audio", output_type="audio")  

# Save the generated speech  
if response['audio'] is not None:  
    import soundfile as sf  
    sf.write("output.wav", response["audio"], 16000)  

print("Generated Text:", response["text"])  
```

## 🌍 **Supported Languages**  

- English (`en`), Hindi (`hi`), Tamil (`ta`), Telugu (`te`), Kannada (`kn`), Malayalam (`ml`), Marathi (`mr`), Bengali (`bn`), Gujarati (`gu`), Punjabi (`pa`), Odia (`or`), Assamese (`as`), Urdu (`ur`).  

## 🎯 **Applications**  

- **Multilingual Virtual Assistants**  
- **Real-time Speech Translation**  
- **Conversational AI for Regional Languages**  
- **Voice-Based Customer Support**  

---