from optibits.eval import calculate_perplexity, evaluate_dataset
from optibits.benchmark import benchmark_latency
from optibits.loader import load_model_and_tokenizer
from optibits.quantization import quantization_configs
from optibits.utils import format_memory_size
from optibits.plotter import plot_comparisons
from datasets import load_dataset
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(model_name, export_path=None, device="cuda"):
    # Take a model, and run it on all types and plot
    quantization_types = []
    memory_sizes = []
    avg_latencies = []
    median_latencies = []
    throughputs = []
    perplexity_scores = []
    mmlu_scores = []

    for quantization_name, quantization_config in quantization_configs.items():
        model, tokenizer = load_model_and_tokenizer(model_name, device=device, quantization_config=quantization_config)

        # memory profiling
        memory_size = model.get_memory_footprint()
        print(f"{quantization_name} model size: {format_memory_size(memory_size)}")

        # # Inspect Model's Data Type
        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.dtype}")  # Check if weights are float32, float16, int8, etc.

        # Benchmarking
        print(f"Running benchmark for {quantization_name}")
        avg_latency, median_latency, throughput = benchmark_latency(model, tokenizer, device=device)
        print(f"avg_latency={avg_latency}, median_latency={median_latency}, throughput={throughput}")

        # ----- EVALUATION ----
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = dataset["text"]  # Extract text samples
        perplexity_score = calculate_perplexity(model, tokenizer, texts[:100])  # Sample 100 texts
        mmlu_score = evaluate_dataset(model, tokenizer, "cais/mmlu", "all", num_samples=128)
        
        # Store data
        quantization_types.append(quantization_name)
        memory_sizes.append(memory_size)
        avg_latencies.append(avg_latency)
        median_latencies.append(median_latency)
        throughputs.append(throughput)
        perplexity_scores.append(perplexity_score)
        mmlu_scores.append(mmlu_score)

    plot_comparisons(model_name, quantization_types, memory_sizes, avg_latencies, median_latencies, throughputs, perplexity_scores, mmlu_scores, export_path)
