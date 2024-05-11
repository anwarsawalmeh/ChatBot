import json
from test import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader





with open('tagging.json', 'r') as f:
    intents = json.load(f)
    
# print(intents)
all_words = []
tags = []
xy = []

ignore_words = ["!", "?", "'", "-","$","&","#", "@", "-", "(", ")", "[","]",".", ","]

for intent in intents['intents']:
    
    # Getting the different pattern/response tags available in intents file
    tag = intent['tag']
    tags.append(tag)
    
    for pattern in intent['pattern']:
        # Tokenize the Patterns
        w = tokenize(pattern)
        
        # Add the tokenized words into the all_words list
        all_words.extend(w)
        
        # Associate the tokenized word with the tag in the XY list
        xy.append((w, tag))

# Stemmer and getting rid of the puncuation. Using List comprehension
all_words = [stem(w) for w in all_words if w not in ignore_words]   
all_words = sorted(set(all_words))

# Sorting the different Tags
tags = sorted(tags)

print(all_words)

x_train = []
y_train = []

for (pattern_tok_sen, tag) in xy:
    bag = bag_of_words(pattern_tok_sen, tag)
    x_train.append(bag)
    
    ind = tags.index(tag)
    y_train.append(ind)

x_train = np.array(x_train)
y_train = np.array(y_train)



class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
        
        
    def __getitem__(self, index):
        return (self.x_data[index], self.y_data[index])
    
    def __len__(self):
        return self.n_samples
    
    dataset = ChatDataset()
    
    