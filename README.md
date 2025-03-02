# OptiBits
OptiBits is a lightweight framework that automates LLM inference efficiency using **quantization, benchmarking, and tuning**—making LLMs **faster and lighter** automatically.

Key features:
- Automates hyperparameter tuning for optimal performance.
- Implements **custom quantization**, including **2-bit quantization**.
- A custom endpoint to quickly visualize
- **Plug & Play**: Directly specify models from Hugging Face.

For example, this is the results of running various quantization parameters GPT2. You can replace this with any model from Hugging Face.

![image](plots/gpt2.png)

# 🔗 Get Started

📌 **GitHub**: [github.com/Gongsta/optibits](https://github.com/Gongsta/optibits)

📌 **Install:**
```sh
pip install optibits
```

📌 **Run:**
```sh
optibits --model mistralai/Mistral-7B
```
This defaults to searching over all quantization strategies.

🚀 **Optimize your LLMs today!**
