# multilingual_chatbot/models/llama.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LlamaModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer (replace with your preferred model)
        model_name = "huggyllama/llama-7b"  # Use your preferred variant
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        ).to(self.device)

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