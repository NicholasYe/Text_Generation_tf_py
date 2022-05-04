# Import TensorFlow and other libraries
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import os
import time

# Read, then decode for py2 compat.
text = open('Embedded_Device.txt', 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print(f'Length of text: {len(text)} characters')

# Take a look at the first 250 characters in text
print(text[:250])

# The unique characters in the file
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')

# convert the strings to a numerical representation.
example_texts = ['abcdefg', 'xyz']
chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
print(chars)

# Now create the tf.keras.layers.StringLookup layer:
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)

# It converts from tokens to character IDs:
ids = ids_from_chars(chars)
print(ids)

# invert this representation and recover human-readable strings from it.
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

# This layer recovers the characters from the vectors of IDs, and returns them as a tf.RaggedTensor of characters:
chars = chars_from_ids(ids)
print(chars)

# use tf.strings.reduce_join to join the characters back into strings. 
tf.strings.reduce_join(chars, axis=-1).numpy()

def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

# convert the text vector into a stream of character indices.
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
for ids in ids_dataset.take(10):
    print(chars_from_ids(ids).numpy().decode('utf-8'))
seq_length = 100

# The batch method lets you easily convert these individual characters to sequences of the desired size.
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

for seq in sequences.take(1):
  print(chars_from_ids(seq))

# It's easier to see what this is doing if you join the tokens back into strings:
for seq in sequences.take(5):
  print(text_from_ids(seq).numpy())

# Here's a function that takes a sequence as input, duplicates, and shifts it to align the input and label for each timestep:
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

split_input_target(list("Tensorflow"))
dataset = sequences.map(split_input_target)

for input_example, target_example in dataset.take(1):
    print("Input :", text_from_ids(input_example).numpy())
    print("Target:", text_from_ids(target_example).numpy())
