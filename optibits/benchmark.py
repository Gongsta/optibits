import time
import torch


def benchmark_latency(
    model, tokenizer, device="cuda", batch_size=8, fp16=False, max_new_tokens=50, runs=4
):
    """Benchmark inference speed in a hardware-agnostic way."""

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if fp16 and device == "cuda":
        model.half()

    text = "Hugging Face models are great because"
    batch = [text] * batch_size
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)

    times = []

    with torch.no_grad():
        for _ in range(runs):
            if device == "cuda":
                torch.cuda.synchronize()  # Ensure previous CUDA ops are done
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                output = model.generate(**inputs, max_new_tokens=max_new_tokens)
                end_time.record()

                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time) / 1000  # Convert ms to seconds
            else:
                start_time = time.perf_counter()
                output = model.generate(**inputs, max_new_tokens=max_new_tokens)
                elapsed_time = time.perf_counter() - start_time

            times.append(elapsed_time)

    generated_texts = tokenizer.batch_decode(output, skip_special_tokens=True)

    print(
        f"Model: {model.name_or_path} | Device: {device} | Batch Size: {batch_size} | FP16: {fp16} | "
        f"First Inference Time: {times[0]:.4f}s | Average Inference Time: {sum(times[1:])/len(times[1:]):.4f}s"
    )
    # print(f"Generated Texts: {generated_texts}\n")
