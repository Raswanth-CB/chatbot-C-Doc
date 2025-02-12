import torch
import os
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset
import deepspeed
import datetime

def setup_distributed():
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, world_size

class TextDataset(Dataset):
    def __init__(self, local_rank):
        self.local_rank = local_rank
        self.output_dir = os.path.expanduser("output/text-to-text")
        os.makedirs(self.output_dir, exist_ok=True)
        self.supported_languages = {
            "as": "Assamese", "bn": "Bengali", "brx": "Bodo", "doi": "Dogri",
            "gom": "Konkani", "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada",
            "ks": "Kashmiri", "mai": "Maithili", "ml": "Malayalam", "mni": "Manipuri",
            "mr": "Marathi", "ne": "Nepali", "or": "Odia", "pa": "Punjabi",
            "sa": "Sanskrit", "sat": "Santali", "sd": "Sindhi", "ta": "Tamil",
            "te": "Telugu", "ur": "Urdu"
        }

    def _distributed_input(self, input_fn):
        if self.local_rank == 0:
            result = input_fn()
            data = torch.tensor(
                [ord(c) for c in result] + [0] * (256 - len(result)), 
                dtype=torch.int16,
                device="cuda"
            )
        else:
            data = torch.zeros(256, dtype=torch.int16, device="cuda")
        
        torch.distributed.broadcast(data, src=0)
        return ''.join([chr(c) for c in data.cpu().tolist() if c != 0])

    def get_input(self):
        def _input_text():
            text = input("\nEnter English text (type 'exit' to quit): ")
            return text.strip()
            
        def _target_lang():
            print("\nSupported languages:")
            for code, lang in self.supported_languages.items():
                print(f"{lang} ({code})")
            
            while True:
                code = input("Enter language code: ").strip().lower()
                if code in self.supported_languages:
                    return code
                print("Invalid code. Try again.")
                
        text = self._distributed_input(_input_text)
        lang = self._distributed_input(_target_lang)
        return text, lang

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.get_input()

def setup_model(model_path, local_rank):
    # DeepSpeed config with NVMe offloading
    ds_config = {
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 3,
            "offload_param": {
                "device": "nvme",
                "nvme_path": "/home/miphi/nvme_offload",  # CHANGE TO YOUR NVME PATH
                "buffer_count": 5,
                "buffer_size": 1e8
            }
        },
        "train_batch_size": 1,
        "steps_per_print": 1,
        "wall_clock_breakdown": False
    }

    # Load model with correct device mapping
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto" if local_rank == 0 else None  # Only rank 0 handles device mapping
    )
    
    # Initialize DeepSpeed engine
    model_engine = deepspeed.initialize(
        model=model,
        config_params=ds_config,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        dist_init_required=True
    )[0]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    return tokenizer, model_engine

def generate_translation(model, tokenizer, dataset, local_rank):
    while True:
        try:
            text, lang = dataset[0]
            if text.lower() == "exit":
                break

            # Prepare input with language code
            input_text = f">>{lang}<< {text}"
            
            # Tokenize with distributed batch handling
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="max_length"
            ).to(model.device)

            # Generate translation
            start_time = time.time()
            outputs = model.generate(
                **inputs,
                num_beams=4,
                max_length=512,
                early_stopping=True,
                temperature=0.7
            )
            
            # Decode and save
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if local_rank == 0:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(
                    dataset.output_dir,
                    f"{lang}_translation_{timestamp}.txt"
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(translation)
                print(f"\nTranslated to {dataset.supported_languages[lang]}:")
                print(f"Output: {translation}")
                print(f"Time: {time.time() - start_time:.2f}s")
                print(f"Saved to: {output_path}\n")

        except Exception as e:
            if local_rank == 0:
                print(f"Error: {str(e)}")
            continue

def main():
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    local_rank, _ = setup_distributed()
    
    # Update model path
    model_path = "/home/miphi/Documents/c-dac/finetune/output_pipeline/model/indictrans2-en-indic-1B"  # Or your local path
    
    tokenizer, model = setup_model(model_path, local_rank)
    dataset = TextDataset(local_rank)
    
    if local_rank == 0:
        print("\nTranslation system ready (DeepSpeed NVMe offloading enabled)")
    
    generate_translation(model, tokenizer, dataset, local_rank)
    
    if local_rank == 0:
        print("Session ended")

if __name__ == "__main__":
    main()