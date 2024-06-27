import pandas as pd
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification

# Load your dataset
dataset_path = "C:\\PythonProjects\\TextFileData\\llama3_dataset.csv"
df = pd.read_csv(dataset_path)

# Example columns assuming 'text' and 'label' are present
texts = df['text'].tolist()
labels = df['label'].tolist()  # Adjust based on your dataset structure

# Initialize the tokenizer and model
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Adjust num_labels as per your task

# Tokenize the inputs
tokenized_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_inputs,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_llama3_model')
