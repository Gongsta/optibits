
import matplotlib.pyplot as plt
import numpy as np

def plot_comparisons(model_name, quantization_types, memory_sizes, avg_latencies, median_latencies, throughputs, perplexity_scores, mmlu_scores, save_path=None):
    plt.style.use("seaborn-v0_8")
    # Create a line plot instead of bar charts
    fig, axs = plt.subplots(5, 1, figsize=(10, 15))
    
    fig.suptitle(model_name, fontsize=16, fontweight='bold')

    # X-axis labels
    x = np.arange(len(quantization_types))

    # Memory Size
    axs[0].plot(quantization_types, memory_sizes, marker='o', linestyle='-', color='b', label="Memory Size (GB)")
    axs[0].set_title("Model Size (GB)")
    axs[0].set_ylabel("GB")
    axs[0].legend()

    # Latency
    axs[1].plot(quantization_types, avg_latencies, marker='o', linestyle='-', color='r', label="Avg Latency")
    axs[1].plot(quantization_types, median_latencies, marker='o', linestyle='-', color='g', label="Median Latency")
    axs[1].set_title("Latency (ms)")
    axs[1].set_ylabel("ms")
    axs[1].legend()

    # Throughput
    axs[2].plot(quantization_types, throughputs, marker='o', linestyle='-', color='purple', label="Throughput")
    axs[2].set_title("Throughput (tokens/sec)")
    axs[2].set_ylabel("Tokens/sec")
    axs[2].legend()

    # Perplexity Score
    axs[3].plot(quantization_types, perplexity_scores, marker='o', linestyle='-', color='orange', label="Perplexity Score")
    axs[3].set_title("Perplexity Score")
    axs[3].set_ylabel("Score")
    axs[3].legend()

    # MMLU Score
    axs[4].plot(quantization_types, mmlu_scores, marker='o', linestyle='-', color='cyan', label="MMLU Score")
    axs[4].set_title("MMLU Score (Accuracy %)")
    axs[4].set_ylabel("Accuracy %")
    axs[4].legend()

    # Adjust layout
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        
    plt.show()
