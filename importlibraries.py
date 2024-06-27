import json
import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          pipeline)
#HF account configuration
config_data = json.load(open("config.json"))
HF_TOKEN = config_data["HF_TOKEN"]

model_name = "meta-llama/Meta-Llama-3-8B"
#Quantization Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

#Loading the Tokenizer and LLM
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token = HF_TOKEN
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='cpu',
    quantization_config=bnb_config,
    token=HF_TOKEN
)

text_generator = pipeline(
    "text_generation",
    model = model_name,
    tokenizer=tokenizer,
    max_new_tokens=128
)
