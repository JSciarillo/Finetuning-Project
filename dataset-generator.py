import pandas as pd
import os

def read_text_files(directory):
    """
    Function to read text from all files in a directory.
    """
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                text = file.read()
                texts.append(text)
    return texts

def generate_pairs_from_files(directory, num_samples):
    """
    Function to generate input/response pairs by reading text from files in a directory.
    """
    texts = read_text_files(directory)
    pairs = []
    for text in texts:
        # Split text into input/response pairs (you can customize this based on your task)
        sentences = text.split('.')  # Split text into sentences
        for i in range(len(sentences) - 1):
            input_text = sentences[i].strip()  # Current sentence
            response_text = sentences[i + 1].strip()  # Next sentence
            pairs.append((input_text, response_text))
    # If number of samples requested is greater than available pairs, repeat pairs
    pairs = pairs * (num_samples // len(pairs)) + pairs[:num_samples % len(pairs)]
    return pairs

def save_dataset_to_csv(pairs, filename):
    """
    Function to save the generated dataset to a CSV file.
    """
    df = pd.DataFrame(pairs, columns=['input_text', 'response_text'])
    df.to_csv(filename, index=False)
    print("Dataset saved to", filename)

if __name__ == "__main__":
    data_directory = "C:\PythonProjects\TextFileData"  # Directory containing text files
    num_samples = 1000  # Number of samples in the dataset
    output_file = "llama3_dataset.csv"  # Output filename

    # Generate input/response pairs from text files
    pairs = generate_pairs_from_files(data_directory, num_samples)

    # Save dataset to CSV
    save_dataset_to_csv(pairs, output_file)
