import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModelForSeq2SeqLM


def run_translation_inference(text, source_lang, target_lang):
    model_path = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True)  
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
    
    # Move model to GPU
    model = model.cuda()
    
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    ).to('cuda')

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            max_length=512,
            num_beams=4,
            early_stopping=True
        )

    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return translation

if __name__ == "__main__":
    text = "Hello, how are you?"
    source_lang = "en"
    target_lang = "hi"

    translated_text = run_translation_inference(text, source_lang, target_lang)
    print(f"Original Text: {text}")
    print(f"Translated Text: {translated_text}")