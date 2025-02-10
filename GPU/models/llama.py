from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
import torch

class LlamaModel:
    def __init__(self, deepspeed_config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            offload_folder="offload"
        )
        
        # Initialize DeepSpeed
        self.ds_engine = deepspeed.init_inference(
            self.model,
            config=deepspeed_config,
            dtype=torch.bfloat16,
            replace_method="auto"
        )
    
    def generate_response(self, prompt, max_length=512):
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        outputs = self.ds_engine.module.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        return self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )