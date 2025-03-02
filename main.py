import argparse
from optibits.model_evaluation import evaluate
# model_names = ["meta-llama/Llama-3.2-3B-Instruct", "gpt2"]
model_names = ["gpt2"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-validate models and export matplotlib figures.")
    # parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model name to cross-validate")
    parser.add_argument("--model", type=str, default="facebook/opt-350m", help="Model name to cross-validate")
    args = parser.parse_args()

    model_name = args.model
    export_path = f"plots/{model_name.replace('/', '-')}"

    evaluate(model_name, export_path)
