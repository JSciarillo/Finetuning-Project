#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
# 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
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

login(token = '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')


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token='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')
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
