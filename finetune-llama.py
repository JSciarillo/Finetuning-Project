# import pandas as pd
# import numpy as np
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from torch.utils.data import Dataset, DataLoader

# class CustomDataset(Dataset):
#     def __init__(self, data_path, tokenizer, max_length=128):
#         self.data = pd.read_csv(data_path)
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         input_text = self.data.iloc[idx]['input_text']
#         response_text = self.data.iloc[idx]['response_text']

#         input_encoding = self.tokenizer(input_text, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
#         response_encoding = self.tokenizer(response_text, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')

#         return input_encoding, response_encoding
# print("Done step 1")
# def fine_tune_ollama_model(dataset_path, model_name, output_dir, batch_size=8, max_length=128, num_epochs=3, learning_rate=1e-5):
#     # Load tokenizer and model
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name)

#     # Load dataset
#     dataset = CustomDataset(dataset_path, tokenizer, max_length=max_length)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     # Define optimizer and loss function
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#     criterion = torch.nn.CrossEntropyLoss()

#     # Training loop
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     for epoch in range(num_epochs):
#         model.train()
#         for batch in dataloader:
#             input_encoding = {key: value.to(device) for key, value in batch[0].items()}
#             response_encoding = {key: value.to(device) for key, value in batch[1].items()}
            
#             # Forward pass
#             outputs = model(**input_encoding, labels=response_encoding["input_ids"])
#             loss = outputs.loss
            
#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

#     # Save fine-tuned model
#     model.save_pretrained(output_dir)
#     tokenizer.save_pretrained(output_dir)
#     print("Fine-tuned model saved to", output_dir)

# if __name__ == "__main__":
#     dataset_path = "C:\\PythonProjects\\TextFileData\\llama3_dataset.csv"  # Path to the input/response pairs dataset
#     model_name = "meta-llama/Meta-Llama-3-8B"  # Model name downloaded through HuggingFace
#     output_dir = "C:\\Users\\jasmi\\Documents\\Fined Tuned Models"  # Directory to save fine-tuned model
#     batch_size = 8
#     max_length = 128
#     num_epochs = 3
#     learning_rate = 1e-5

#     # Fine-tune OLLAMA model
#     fine_tune_ollama_model(dataset_path, model_name, output_dir, batch_size, max_length, num_epochs, learning_rate)

# print("Finetuning completed")

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data.iloc[idx]['input_text']
        response_text = self.data.iloc[idx]['response_text']

        input_encoding = self.tokenizer(input_text, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        response_encoding = self.tokenizer(response_text, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')

        return input_encoding, response_encoding

def fine_tune_llama3(dataset_path, model_name, output_dir, batch_size=8, max_length=128, num_epochs=3, learning_rate=1e-5):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load dataset
    dataset = CustomDataset(dataset_path, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            input_encoding = {key: value.to(device) for key, value in batch[0].items()}
            response_encoding = {key: value.to(device) for key, value in batch[1].items()}
            
            # Forward pass
            outputs = model(**input_encoding, labels=response_encoding["input_ids"])
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")

    # Save fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")

if __name__ == "__main__":
    dataset_path = "C:\\PythonProjects\\TextFileData\\llama3_dataset.csv"
    model_name = "meta-llama/Meta-Llama-3-8B"  # Adjust to the correct model identifier or path
    output_dir = "C:\\Users\\jasmi\\Documents\\FineTunedModels"  # Adjust as needed
    batch_size = 8
    max_length = 128
    num_epochs = 3
    learning_rate = 1e-5

    fine_tune_llama3(dataset_path, model_name, output_dir, batch_size, max_length, num_epochs, learning_rate)
