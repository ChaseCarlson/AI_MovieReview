import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

word_count = 10000

word_index = data.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def get_data():
	return data

def get_word_index():
	return word_index

def get_reverse_word_index():
	return reverse_word_index

def load_data():
	return data.load_data(num_words=word_count)

def decode_review(arr):
	return " ".join([reverse_word_index.get(i, "?") for i in arr])

def encode_review(text):
	word_array = text.split(" ")
	index_array = []
	for i, word in enumerate(word_array):
		widx = word_index.get(word, word_index["<UNK>"])
		if widx < word_count:
			word = word.lower()
			index_array.append(widx)
	return index_array

def get_review_string(value):
	return "Positive" if value > 0.5 else "Negative"