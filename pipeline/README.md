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
