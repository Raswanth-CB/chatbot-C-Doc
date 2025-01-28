# Pipeline Execution
## 1. Process Audio Input to Tamil Text
    python main.py \
    --input speech.mp3 \
    --input-type audio \
    --output output.txt \
    --output-lang ta \
    --output-format text
## 2. Process Hindi Text to Telugu Audio

    python main.py \
    --input input.txt \
    --input-type text \
    --output output.wav \
    --output-lang te \
    --output-format audio
## 3. Start RL Training

    python rl_training.py



## Key Features
End-to-End Multimodal Processing: Handles both text and audio inputs/outputs

Language Agnostic: Supports 22 Indian languages + English

Efficient Resource Management: NVMe offloading via DeepSpeed Zero-3

Reinforcement Learning: Customizable reward mechanism for Llama-8B

Production-Ready: Modular architecture with proper error handling
