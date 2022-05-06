import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
import string
import warnings
warnings.filterwarnings("ignore")

# import matplotlib.pyplot as plt
# %matplotlib inline
from tqdm import tqdm

import renom as rm
from renom.cuda import set_cuda_active
set_cuda_active(True)

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

text = load_doc('paper_ch.txt')

# turn a doc into clean tokens
def clean_doc(doc):
    # replace '--' with a space ' '
    doc = doc.replace('--', ' ')
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]
    return tokens

tokens = clean_doc(text)
# print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))

length = 50 + 1
look_back = 50
sequences = list()
for i in range(length, len(tokens)):
    # select sequence of tokens
    seq = tokens[i-length:i]
    # convert into a line
    line = ' '.join(seq)
    # store
    sequences.append(line)
print('Total Sequences: %d' % len(sequences))

def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w', encoding='utf-8')
    file.write(data)
    file.close()

out_filename = 'wonderland_sequences.txt'
save_doc(sequences, out_filename)