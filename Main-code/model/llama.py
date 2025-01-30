import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed
from typing import Optional, Dict, Any


class LlamaModel:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        deepspeed_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize LLaMA model with DeepSpeed optimization."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with DeepSpeed optimization
        model = AutoModelForCausalLM.from_pretrained(model_name)

        if deepspeed_config:
            self.model = deepspeed.initialize(
                model=model,
                config=deepspeed_config
            )[0]
        else:
            self.model = model.cuda()

    def generate_response(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate response from LLaMA model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated response text
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()

    def __call__(self, *args, **kwargs):
        return self.generate_response(*args, **kwargs)
