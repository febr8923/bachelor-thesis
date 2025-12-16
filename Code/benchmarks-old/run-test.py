import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_basic_cpu_inference(model_name: str, text: str):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Tokenize input text
    input_batch = tokenizer(text, return_tensors='pt')

    # Load model on CPU with flash attention and mixed precision disabled
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="cpu",
        bf16=False,
        fp16=False,
        fp32=True
    )

    # Ensure inputs are on CPU
    input_batch = {k: v.to('cpu') for k, v in input_batch.items()}

    model.eval()
    with torch.no_grad():
        output = model.generate(**input_batch, max_length=20)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated text:", generated_text)

if __name__ == "__main__":
    model_name = "Qwen/Qwen-7B"  # replace with any appropriate model name
    text = "Hello, world!"
    run_basic_cpu_inference(model_name, text)
