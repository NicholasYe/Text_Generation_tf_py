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

text = load_doc('wonderland.txt')

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

seq = load_doc('wonderland_sequences.txt')
lines = seq.split('\n')

def Tokenizer(lines):
    tokenizer = LabelEncoder()
    seq_gen = tokenizer.fit(tokens)
    sequences = []
    for i in range(len(lines)):
        temp = lines[i].split(' ')
        sequences.append(seq_gen.transform(temp))
    return sequences,seq_gen

sequences,tokenizer = Tokenizer(lines)

vocab_size = len(tokenizer.classes_) # needed to check if 1 shouldbe added or not

sequences = np.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = LabelBinarizer().fit_transform(y)
seq_length = X.shape[1]

# Loading pretrained embedding layer

embeddings_index = dict()
f = open('E:/WPS_Sync_Files/Tensorflow_Database/glove.6B/glove.6B.50d.txt','r',encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((vocab_size, 50))
token_obj = dict(zip(tokenizer.classes_, tokenizer.transform(tokenizer.classes_)))
for word, i in token_obj.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

class NlpTextGen(rm.Model):
    def __init__(self):
        self._embed = rm.Embedding(input_size=vocab_size,output_size=60)
        self._lstm1 = rm.Lstm(100)
        self._lstm2 = rm.Lstm(100)
        self._dnse1 = rm.Dense(100)
        self._dnse2 = rm.Dense(vocab_size)

    # Definition of forward calculation.
    def forward(self, x):
        out = self._embed(x)
        out = self._lstm1(out)
        out = self._lstm2(out)
        out = self._dnse1(out)
        out = self._dnse2(rm.relu(out))
        return out

model = NlpTextGen()
model._embed.params['w'] = rm.Variable(embedding_matrix)
model._embed.set_prevent_update(True)
model.set_models(inference=False)

# params
batch_size = 256
epoch = 20
optimizer = rm.Adam()
# learning curves
learning_curve = []

if not os.path.exists("trained_models_pretrained_word_embeddings"):
    os.makedirs("trained_models_pretrained_word_embeddings")

model_num = 0
# train loop
for i in range(epoch):
    bar = tqdm(range(len(X) // batch_size))
    # perm is for getting batch randomly
#     perm = np.random.permutation(n_patterns)
    train_loss = 0
    for j in range(len(X) // batch_size):
        batch_x = X[j*batch_size : (j+1)*batch_size]
        batch_y = y[j*batch_size : (j+1)*batch_size]
        # Forward propagation
        l = 0
        z = 0
        with model.train():
            for t in range(look_back):
                z = model(batch_x[:,t].reshape(len(batch_x),-1))
                l = rm.softmax_cross_entropy(z, batch_y)
            model.truncate()
        l.grad().update(optimizer)
        train_loss += l.as_ndarray()
        bar.set_description("epoch {:03d} train loss:{:6.4f}".format(i, float(l.as_ndarray())))
        bar.update(1)

    train_loss = train_loss / (len(X) // batch_size)
    learning_curve.append(train_loss)

#     print('epoch:{}, train loss:{}'.format(i, train_loss))
    bar.set_description("epoch {:03d} avg loss:{:6.4f}".format(i, float(train_loss)))
    bar.update(0)
    bar.refresh()
    bar.close()
    model.save('trained_models_pretrained_word_embeddings/trained_model_'+str(i)+'.h5')
    model_num =  i

model.load('trained_models_pretrained_word_embeddings/trained_model_'+str(model_num)+'.h5')
model.set_models(inference=True)
model.set_prevent_update(True)

start = np.random.randint(0, len(X)-1)
pattern = list(X[start])

print("Input:")
print('\'',' '.join(tokenizer.inverse_transform(pattern)),'\'')

sentence_list=[]
print('\n\nPlease wait!! generating sentences...')
for i in range(100):
    for t in range(look_back):
        pred = rm.softmax(model(pattern[t].reshape(1,1))).as_ndarray()
        pred = np.argmax(pred)
    model.truncate()
    pattern = np.delete(pattern, 0)
    pattern = np.append(pattern, pred)
    sentence_list.append(pred)

print('\n\nOutput:')
print('\'',' '.join(tokenizer.inverse_transform(sentence_list)),'\'')

