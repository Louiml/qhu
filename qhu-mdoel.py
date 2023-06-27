import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Tokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0

    def tokenize(self, text):
        tokens = text.split()
        return tokens

    def detokenize(self, tokens):
        text = ' '.join(tokens)
        return text

    def build_vocab(self, texts):
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                if token not in self.token_to_id:
                    self.token_to_id[token] = self.vocab_size
                    self.id_to_token[self.vocab_size] = token
                    self.vocab_size += 1

    def get_token_id(self, token):
        return self.token_to_id[token]

    def get_token_from_id(self, token_id):
        return self.id_to_token[token_id]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.positional_encoding = self.generate_positional_encoding(d_model, max_len)

    def generate_positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def forward(self, x):
        x = x + self.positional_encoding[:x.size(0), :]
        return self.dropout(x)

def tokenize(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    token_ids = [tokenizer.get_token_id(token) for token in tokens]
    return token_ids

vocab_size = 3285 
embedding_dim = 128
num_heads = 4
num_layers = 2
tokenizer = Tokenizer() 
memory = None

dataset_file = "dataset.txt"
with open(dataset_file, "r") as file:
    dataset = file.readlines()

tokenizer.build_vocab(dataset)
torch.save(tokenizer, "tokenizer.pth")