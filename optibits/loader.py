from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_model_and_tokenizer(model_name, device="cuda", quantization_config=None):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_cache=False)
    if quantization_config is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False).to(device).eval()
    elif quantization_config == "FP16":
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=False).to(device).eval()
    elif quantization_config == "BFLOAT16":
        model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16, use_cache=False).to(device).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, use_cache=False).eval()

    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS as padding token

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
