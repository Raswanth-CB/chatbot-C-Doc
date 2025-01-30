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

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    return local_rank, world_size

class TextDataset(Dataset):
    def __init__(self, local_rank):
        self.local_rank = local_rank
        self.output_dir = os.path.expanduser("output/text-to-text")
        os.makedirs(self.output_dir, exist_ok=True)

    def get_input_text(self):
        if self.local_rank == 0:
            text = input("\nEnter the text you want to translate (type 'exit' to quit): ")
            text_tensor = torch.tensor([ord(c) for c in text] + [0] * (1024 - len(text)), dtype=torch.long).cuda()
        else:
            text_tensor = torch.zeros(1024, dtype=torch.long).cuda()

        torch.distributed.broadcast(text_tensor, src=0)
        return ''.join([chr(c.item()) for c in text_tensor.cpu() if c != 0])

    def get_target_language(self):
        supported_languages = {
            "as": "Assamese", "bn": "Bengali", "brx": "Bodo", "doi": "Dogri",
            "gom": "Konkani", "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada",
            "ks": "Kashmiri", "mai": "Maithili", "ml": "Malayalam", "mni": "Manipuri",
            "mr": "Marathi", "ne": "Nepali", "or": "Odia", "pa": "Punjabi",
            "sa": "Sanskrit", "sat": "Santali", "sd": "Sindhi", "ta": "Tamil",
            "te": "Telugu", "ur": "Urdu"
        }

        if self.local_rank == 0:
            print("\nSupported Indian languages:")
            for code, lang in supported_languages.items():
                print(f"{lang} ({code})")

            while True:
                lang_code = input("Enter the target language code (e.g., 'ta' for Tamil): ").strip().lower()
                if lang_code in supported_languages:
                    break
                print(f"Invalid language code. Please choose from: {', '.join(supported_languages.keys())}")

            lang_tensor = torch.tensor([ord(c) for c in lang_code] + [0] * (10 - len(lang_code)), dtype=torch.long).cuda()
        else:
            lang_tensor = torch.zeros(10, dtype=torch.long).cuda()

        torch.distributed.broadcast(lang_tensor, src=0)
        return ''.join([chr(c.item()) for c in lang_tensor.cpu() if c != 0])

    def get_output_path(self, index):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_dir, f"translation_output_{timestamp}_{index}.txt")

    def __len__(self):
        return 1

    def __getitem__(self, index):
        text = self.get_input_text()
        if text.lower() == "exit":
            raise StopIteration("Exit command received.")

        target_lang = self.get_target_language()
        return text, target_lang

def setup_model(model_name, local_rank):
    print(f"Rank {local_rank}: Loading model and tokenizers...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Use float16 for better performance
    ).to(f"cuda:{local_rank}")
    
    # Initialize DeepSpeed inference engine
    ds_engine = deepspeed.init_inference(
        model=model,
        mp_size=1,
        dtype=torch.float16,
        replace_method="auto",
        replace_with_kernel_inject=True
    )
    
    model = ds_engine.module

    print(f"Rank {local_rank}: Model setup complete with DeepSpeed inference engine.")
    return tokenizer, model

def generate_translation(model, tokenizer, dataset, local_rank):
    while True:
        try:
            text, target_lang = dataset[0]

            # Format input text with language token
            input_text = f">>{target_lang}<< {text}"
            print(f"Rank {local_rank}: Processing input: {input_text}")

            # Tokenize inputs
            text_inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(f"cuda:{local_rank}")

            # Generate translation
            start_time = time.time()
            with torch.no_grad():
                translation = model.generate(
                    input_ids=text_inputs.input_ids,
                    attention_mask=text_inputs.attention_mask,
                    num_beams=4,
                    max_length=512,
                    early_stopping=True,
                    repetition_penalty=2.0,
                    no_repeat_ngram_size=2,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9
                )

            # Decode and save translation
            translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)

            if local_rank == 0:
                output_path = dataset.get_output_path(0)
                with open(output_path, "w", encoding='utf-8') as f:
                    f.write(translated_text)
                print(f"\nTranslation: {translated_text}")
                print(f"Saved to: {output_path}")
                print(f"Translation time: {time.time() - start_time:.2f} seconds")

        except StopIteration:
            break
        except Exception as e:
            if local_rank == 0:
                print(f"Error: {str(e)}")
                print(f"Stack trace:", exc_info=True)
            continue

def main():
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    
    local_rank, world_size = setup_distributed()
    model_name = "/home/miphi/Documents/c-dac/finetune/output_pipeline/model/finetuned"

    print("\nInitializing model with DeepSpeed inference engine...")
    tokenizer, model = setup_model(model_name, local_rank)
    dataset = TextDataset(local_rank)

    if local_rank == 0:
        print("\nStarting IndicTrans2 Translation with DeepSpeed. Enter 'exit' to quit.")

    generate_translation(model, tokenizer, dataset, local_rank)

    if local_rank == 0:
        print("\nTranslation session complete!")

if __name__ == "__main__":
    main()
