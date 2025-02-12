import torch
import os
import soundfile as sf
import time
import numpy as np
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from torch.utils.data import Dataset
import deepspeed
import datetime
import noisereduce as nr

def setup_distributed():
    instance_number = 1
    base_port = 29600
    instance_port = base_port + instance_number

    os.environ['MASTER_PORT'] = str(instance_port)
    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    print(f"Initializing distributed environment: local_rank={local_rank}, world_size={world_size}")

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print(f"Distributed setup complete: local_rank={local_rank}, world_size={world_size}")

    return local_rank, world_size

class TextDataset(Dataset):
    def __init__(self, local_rank):
        self.local_rank = local_rank
        self.output_dir = os.path.expanduser("output/text-to-speech")  # Use a directory in your home folder
        os.makedirs(self.output_dir, exist_ok=True)

    def get_input_text(self):
        if self.local_rank == 0:
            text = input("\nEnter the text you want to convert to speech (type 'exit' to quit): ")
            text_tensor = torch.tensor([ord(c) for c in text] + [0] * (1024 - len(text)), dtype=torch.long).cuda()
        else:
            text_tensor = torch.zeros(1024, dtype=torch.long).cuda()

        print(f"Rank {self.local_rank}: Broadcasting input text...")
        torch.distributed.broadcast(text_tensor, src=0)
        print(f"Rank {self.local_rank}: Input text broadcast complete.")

        return ''.join([chr(c.item()) for c in text_tensor.cpu() if c != 0])

    def get_description(self):
        if self.local_rank == 0:
            description = input("Enter voice description (press Enter for default): ") or \
                "A female speaker with a British accent delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."
            desc_tensor = torch.tensor([ord(c) for c in description] + [0] * (1024 - len(description)), dtype=torch.long).cuda()
        else:
            desc_tensor = torch.zeros(1024, dtype=torch.long).cuda()

        print(f"Rank {self.local_rank}: Broadcasting description...")
        torch.distributed.broadcast(desc_tensor, src=0)
        print(f"Rank {self.local_rank}: Description broadcast complete.")

        return ''.join([chr(c.item()) for c in desc_tensor.cpu() if c != 0])

    def get_output_path(self, index):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_dir, f"tts_output_{timestamp}_{index}.wav")

    def __len__(self):
        return 1

    def __getitem__(self, index):
        text = self.get_input_text()
        if text.lower() == "exit":
            raise StopIteration("Exit command received.")

        description = self.get_description()
        return text, description

def setup_model(model_name, local_rank):
    print(f"Rank {local_rank}: Loading model and tokenizers...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    description_tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure the tokenizer has a unique padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add a unique padding token

    if description_tokenizer.pad_token is None:
        description_tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add a unique padding token

    print(f"Rank {local_rank}: Converting model to float16 and moving to device {local_rank}...")
    model = model.to(torch.float16).to(local_rank)

    for param in model.parameters():
        param.data = param.data.to(torch.float16)

    print(f"Rank {local_rank}: Model setup complete.")
    return tokenizer, description_tokenizer, model


def generate_speech(model, tokenizer, description_tokenizer, dataset, local_rank):
    while True:
        try:
            text, description = dataset[0]

            # Tokenize inputs
            description_inputs = description_tokenizer(
                description,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(local_rank)

            text_inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(local_rank)

            # Debug input shapes
            print("Description Input IDs Shape:", description_inputs.input_ids.shape)
            print("Text Input IDs Shape:", text_inputs.input_ids.shape)

            # Generate audio
            start_time = time.time()
            with torch.no_grad():
                generation = model.generate(
                    input_ids=description_inputs.input_ids,
                    attention_mask=description_inputs.attention_mask,
                    prompt_input_ids=text_inputs.input_ids,
                    prompt_attention_mask=text_inputs.attention_mask,
                    num_beams=1,  # Disable beam search
                    do_sample=True,  # Enable sampling
                    temperature=0.7,  # Adjust for randomness
                    top_p=0.9,  # Nucleus sampling
                    top_k=50,  # Limit sampling pool
                    repetition_penalty=1.5,  # Avoid repetition
                    use_cache=True,  # Enable caching
                    max_length=512  # Adjust based on input length
                )

            # Debug generation tensor
            print("Generation Tensor Shape:", generation.shape)
            print("Generation Tensor Values:", generation)

            # Process and save the audio
            if local_rank == 0:
                audio_arr = generation.cpu().numpy().squeeze().astype(np.float32)
                print("Audio Array Shape:", audio_arr.shape)
                print("Audio Array Max Value:", np.abs(audio_arr).max())

                # Normalize audio to prevent clipping
                if np.abs(audio_arr).max() > 0:
                    audio_arr = audio_arr / np.abs(audio_arr).max()

                # Reduce noise
                audio_arr_cleaned = nr.reduce_noise(y=audio_arr, sr=model.config.sampling_rate, stationary=True)

                # Save the audio
                output_path = dataset.get_output_path(0)
                sf.write(output_path, audio_arr_cleaned, model.config.sampling_rate)
                print(f"\nGenerated audio saved to: {output_path}")
                print(f"Generation time: {end_time - start_time:.2f} seconds")

        except StopIteration:
            break
        except Exception as e:
            if local_rank == 0:
                print(f"Error: {str(e)}")
            continue
        
def main():
    # Set environment variable for protobuf implementation
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    local_rank, world_size = setup_distributed()
    model_name = "model/indic-parler-tts-pretrained"  # Updated model name

    print("\nInitializing model and tokenizers...")
    tokenizer, description_tokenizer, model = setup_model(model_name, local_rank)
    dataset = TextDataset(local_rank)

    if local_rank == 0:
        print("\nStarting Indic Parler TTS. Enter 'exit' to quit.")

    generate_speech(model, tokenizer, description_tokenizer, dataset, local_rank)

    if local_rank == 0:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()