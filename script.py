from huggingface_hub import snapshot_download

# https://huggingface.co/docs/huggingface_hub/en/guides/download
# model_id = "openai-community/gpt2"
# model_id = "TinyLlama/TinyLlama_v1.1"
model_id = "distilbert/distilgpt2"
# Downlaod the entire repository
# https://github.com/ggml-org/llama.cpp/discussions/2948
snapshot_download(repo_id=model_id, local_dir=f"models/{model_id.split('/')[1]}")
