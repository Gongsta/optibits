from optibits.benchmark import benchmark_latency
from optibits.loader import load_model_and_tokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_names = ["gpt2", "llama3"]

for model_name in model_names:
    model, tokenizer = load_model_and_tokenizer(model_names, device=device)
    benchmark_latency(model, tokenizer, device=device)
