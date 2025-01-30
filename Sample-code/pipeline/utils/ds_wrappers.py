from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoModelForSeq2SeqLM
)
import yaml
import deepspeed

def load_model(model_type):
    with open("configs/models.yaml", "r") as f:
        config = yaml.safe_load(f)["models"]
    
    model_config = config[model_type]
    
    if model_type == "llama":
        model = AutoModelForCausalLM.from_pretrained(
            model_config["name"],
            device_map="auto"
        )
    elif model_type == "whisper":
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_config["name"],
            device_map="auto"
        )
    elif model_type == "indic_trans":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_config["name"],
            device_map="auto"
        )
    
    return deepspeed.init_inference(
        model,
        config_file="configs/ds_offload.json",
        dtype="fp16",
        replace_with_kernel_inject=True
    )
