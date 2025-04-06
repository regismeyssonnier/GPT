# -*- coding: latin-1 -*-

from datasets import load_dataset
from transformers import GPT2TokenizerFast
from datasets import Dataset
import json
import torch
from torch.utils.data import Dataset as TorchDataset
from collections import defaultdict
from transformers import AutoTokenizer

# Ton vocabulaire simple pour des calculs
#VOCAB = {ch: i for i, ch in enumerate("0123456789+-*/=abcdefghiklmnopqrstuvwxyz ")}
#VOCAB = {ch: i for i, ch in enumerate("0123456789 plus minus times divided equal remainder ")}
VOCAB = {str(i): i for i in range(10001)}
VOCAB.update({
    'plus': 10001,
    'minus': 10002,
    'times': 10003,
    'divided': 10004,
    'by': 10005,
    'equal': 10006,
    '<pad>': 10007
})
VOCAB_SIZE = 30522 #len(VOCAB)

# Inverser le vocab pour d�codage plus tard si besoin
INV_VOCAB = {i: ch for ch, i in VOCAB.items()}

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_words(text, max_length=8):
    return [VOCAB.get(word, VOCAB['<pad>']) for word in text.split()]

# Remplace cette fonction par un tokeniseur Hugging Face ou ton propre tokenizer
def tokenize_words2(text, max_length):
    # Initialize the tokenizer for a pre-trained BERT model
        
    # Tokenize the input text with truncation and padding
    tokens = tokenizer.encode(text, truncation=True, padding="max_length", max_length=max_length, add_special_tokens=True)
    
    # Print out the tokens for debugging purposes
    #print(f"Original Text: {text}")
    #print(f"Tokenized Output: {tokens}")
    
    return tokens

# Adaptation de la fonction
def get_tokenized_dataset_perso_calc_w(dataset_name="calculsp.json", max_length=32):
    with open(dataset_name, 'r') as f:
        raw_data = json.load(f)

    dataset = [{"input_ids": tokenize_words2(item["text"], max_length)} for item in raw_data]

    # Conversion en Dataset HuggingFace (optionnelle)
    hf_dataset = Dataset.from_dict({
        "input_ids": [item["input_ids"] for item in dataset]
    })

    # Format torch
    hf_dataset.set_format(type="torch", columns=["input_ids"])
    return hf_dataset, tokenizer


# Tokenizer perso
def tokenize_text(text, max_length=32):
    token_ids = [VOCAB.get(ch, VOCAB[" "]) for ch in text[:max_length]]
    padding = [VOCAB[" "]] * (max_length - len(token_ids))
    return token_ids + padding

# Adaptation de la fonction
def get_tokenized_dataset_perso_calc(dataset_name="calculsp.json", max_length=32):
    with open(dataset_name, 'r') as f:
        raw_data = json.load(f)

    dataset = [{"input_ids": tokenize_text(item["text"], max_length)} for item in raw_data]

    # Conversion en Dataset HuggingFace (optionnelle)
    hf_dataset = Dataset.from_dict({
        "input_ids": [item["input_ids"] for item in dataset]
    })

    # Format torch
    hf_dataset.set_format(type="torch", columns=["input_ids"])
    return hf_dataset

def get_tokenized_dataset(dataset_name="wikipedia", subset="20220301.simple", max_length=512):
    # Charger le dataset
    dataset = load_dataset(dataset_name, subset)

    # Initialiser le tokenizer GPT-2
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Ajouter le pad_token (utilisation du EOS token comme padding)
    tokenizer.pad_token = tokenizer.eos_token

    # Fonction de tokenisation
    def tokenize_function(examples):
        tokenizer_output = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        return {"input_ids": tokenizer_output["input_ids"]}

    # Appliquer la tokenisation
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=1, remove_columns=[col for col in dataset['train'].column_names if col not in ['text']])
    print(tokenized_datasets['train'].column_names)  

    # Convertir en tensors PyTorch
    tokenized_datasets.set_format(type="torch", columns=["input_ids"])

    return tokenized_datasets

def get_tokenized_dataset_regis(dataset_name='regis.json', max_length=512):
    # Charger le dataset JSON
    with open(dataset_name, 'r') as f:
        raw_data = json.load(f)

    # Créer un Dataset HuggingFace à partir des textes
    hf_dataset = Dataset.from_dict({
        "text": [item["text"] for item in raw_data]
    })

    # Initialiser le tokenizer GPT-2
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Ajouter un pad_token

    # Fonction de tokenisation
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    # Appliquer la tokenisation
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)

    # Convertir en tensors PyTorch
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return tokenized_dataset

def get_tokenized_dataset_perso(dataset_name="calculs.json", max_length=512):
    # Charger le fichier JSON dans une variable Python
    with open(dataset_name, 'r') as f:
        dataset = json.load(f)

    # Convertir en format Dataset de Hugging Face
    dataset = Dataset.from_dict({"text": [item["text"] for item in dataset]})

    # Initialiser le tokenizer GPT-2
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Ajouter le pad_token (utilisation du EOS token comme padding)
    tokenizer.pad_token = tokenizer.eos_token

    # Fonction de tokenisation
    def tokenize_function(examples):
        tokenizer_output = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        return {"input_ids": tokenizer_output["input_ids"]}

    # Appliquer la tokenisation
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Convertir en tensors PyTorch
    tokenized_datasets.set_format(type="torch", columns=["input_ids"])

    return tokenized_datasets

if __name__ == '__main__':
    
    tokenized_datasets = get_tokenized_dataset()

    print("Dataset tokenise et pret !")
