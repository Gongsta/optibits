from transformers import BitsAndBytesConfig
import torch

non_quantized_config = BitsAndBytesConfig(
    load_in_4bit=False,
    load_in_8bit=False,
)

quantization_8_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

# 4-bit quantization
# https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf#scrollTo=pDrUsbj66MGl
quantization_4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# nf4 comes from QLora paper
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

quantization_configs = {
    "FP32": None, 
    "FP16": "FP16",
    "BFLOAT16": "BFLOAT16",
    "INT8": quantization_8_config,
    "INT4": quantization_4_config,
    "NF4": nf4_config,
    "DOUBLE_QUANT": double_quant_config,
}
