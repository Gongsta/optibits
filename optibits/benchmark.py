import time
import torch


def benchmark_latency(
    model, tokenizer, device="cuda", batch_size=1, runs=8, warmup_runs=2):
    """Benchmark inference speed in a hardware-agnostic way."""

    text = "Hugging Face models are great because"
    batch = [text] * batch_size
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)

    # Warm-up runs to eliminate startup overhead
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model.generate(**inputs);

    times = []
    total_tokens_generated = 0

    with torch.no_grad():
        for _ in range(runs):
            if device == "cuda":
                torch.cuda.current_stream().synchronize()
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                output = model.generate(**inputs)
                end_time.record()

                torch.cuda.current_stream().synchronize()
                elapsed_time = start_time.elapsed_time(end_time) / 1000  # Convert ms to seconds
            else:
                start_time = time.perf_counter()
                output = model.generate(**inputs)
                elapsed_time = time.perf_counter() - start_time

            times.append(elapsed_time)
            total_tokens_generated += sum(len(seq) for seq in output)

    avg_latency = sum(times) / len(times)
    median_latency = sorted(times)[len(times) // 2]
    throughput = total_tokens_generated / sum(times)

    print(
        f"Model: {model.name_or_path} | Device: {device} | Batch Size: {batch_size} | "
        f"Median Latency: {median_latency:.4f}s | Average Latency: {avg_latency:.4f}s | "
        f"Throughput: {throughput:.2f} tokens/sec"
    )

    return avg_latency, median_latency, throughput
