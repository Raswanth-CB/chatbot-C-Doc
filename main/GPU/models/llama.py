# multilingual_chatbot/models/llama.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

class LlamaModel:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-enabled GPU is required but not available")
            
        self.device = torch.device("cuda")
        
        # 4-bit quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained("/home/miphi/raswanth/model/Llama-3.1-8B")
        self.model = AutoModelForCausalLM.from_pretrained(
            "/home/miphi/raswanth/model/Llama-3.1-8B",
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )

    def generate_response(self, prompt, max_length=200):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    