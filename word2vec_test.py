import word2vec
import sys
import os

'''
Usage: python get_vocab.py /path/to/vocab.bin
'''
w2v_file = 'vocab.bin'
model = word2vec.load(w2v_file)

vocab = model.vocab