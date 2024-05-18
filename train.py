import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from test import tokenize, stem, bag_of_words
from model import NerualNet

# Load the intents JSON file
with open('tagging.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

ignore_words = ["!", "?", "'", "-", "$", "&", "#", "@", "-", "(", ")", "[", "]", ".", ","]

# Process each intent
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    
    for pattern in intent['pattern']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Stem and ignore punctuation
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))

# Sort tags
tags = sorted(tags)

# Print to verify
print("All words:", all_words)
print("Tags:", tags)

x_train = []
y_train = []

# Create training data
for (pattern_tok_sen, tag) in xy:
    bag = bag_of_words(pattern_tok_sen, all_words)  # Use all_words here
    x_train.append(bag)
    
    ind = tags.index(tag)
    y_train.append(ind)

# Convert to NumPy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

# Verify shapes
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
        
    def __getitem__(self, index):
        return (self.x_data[index], self.y_data[index])
    
    def __len__(self):
        return self.n_samples

# Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(x_train[0])  # This should be the length of all_words

print(input_size, len(all_words))
print(output_size, tags)

# Create dataset and data loader
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Initialize model
model = NerualNet(input_size, hidden_size, output_size)
    
    