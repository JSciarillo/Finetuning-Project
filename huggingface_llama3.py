from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForSeq2SeqLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

tokenizer.save_pretrained("C:\\Users\\jasmi\\Documents\\Hugging_Face_Llama3")
model.save_pretrained("C:\\Users\\jasmi\\Documents\\Hugging_Face_Llama3")

