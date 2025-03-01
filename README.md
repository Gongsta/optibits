# etched-hackathon

Goal: Feed in a hugging-face model. Automatically run conversion into GGUF. Then, find optimal
parameters.

Update: switching this to python, building a universal benchmarking suite. And then, once you
have your benchmark, be able to evaluate how changes to the model affect output.

# OptiBits: Auto-Optimizing LLM Inference

### What is it?

OptiBits automates LLM inference efficiency using **quantization, benchmarking, and tuning**—making LLMs **faster and lighter** automatically.

### Key Features:

- ✅ Auto-applies **8-bit & 4-bit quantization** for speedup
- New floating point quantization that acheves. 1.58bit.

✅ **Benchmarks latency vs. quantization level**
✅ **Plots optimization trade-offs** for easy visualization
✅ Designed for **LLM inference efficiency** in real-world applications

### \*\*Step 3: LIVE DEMO Plan

Instead of slides, show:

1. **Before & After Quantization** (print inference times)
2. **Benchmark Graph** (show the matplotlib plot)
3. **Explain why auto-quantization matters** (less memory, faster response!)

---

### **Final Push – What to Focus on**

💥 **MOST IMPORTANT:**  
✅ Get auto-quantization & benchmarking running  
✅ Show **a graph of performance trade-offs**  
✅ Have **a quick, clear README**

💡 Need anything last-minute, or should I tweak something?

### Quick Start:

```bash
pip install -e .
```

Specify your model (pytorch model).

We also ran this on Llama 70B. GPT2
