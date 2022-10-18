import argparse
import json
import sys

from pprint import pprint

import numpy as np
from numpy import load

from syntok.tokenizer import Tokenizer
from hmm import build_vocab2idx, create_dictionaries, create_transition_matrix, create_emission_matrix, initialize, viterbi_forward, viterbi_backward
from hmm import training_data
from utils import processing

CORPUS_PATH = "data/WSJ_02-21.pos"
alpha = 0.001

def parse_argument():
    parser = argparse.ArgumentParser(description='Predict Part of Speech Tags')
    parser.add_argument('--sent', help='Enter your sentence.')
    return parser.parse_args()

def predict(sample):
    #~ pprint(sample)
    #~ tokens = word_tokenize(sample)
    tok = Tokenizer()
    tokens = [token.value for token in tok.tokenize(sample)]
    #~ pprint(tokens)
    vocab2idx = build_vocab2idx(CORPUS_PATH)
    #~ file = open('vocab.pkl', 'rb')
    #~ vocab2idx = pickle.load(file)
    #~ file.close()
    #~ pprint(vocab2idx)
    
    prep_tokens = processing(vocab2idx, tokens)
    training_corpus = training_data(CORPUS_PATH)
    emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab2idx)
    states = sorted(tag_counts.keys())
    alpha = 0.001
    A = create_transition_matrix(transition_counts, tag_counts, alpha)
    B = create_emission_matrix(emission_counts, tag_counts, list(vocab2idx), alpha)
    #~ A = load('A.npy')
    #~ B = load('B.npy')
    best_probs, best_paths = initialize(A, B, tag_counts, vocab2idx, states, prep_tokens)
    best_probs, best_paths = viterbi_forward(A, B, prep_tokens, best_probs, best_paths, vocab2idx)
    pred = viterbi_backward(best_probs, best_paths, states)

    res = []
    for tok, tag in zip(prep_tokens[:-1], pred[:-1]):
        res.append((tok, tag))
    print(res)

if __name__ == "__main__":
    args = parse_argument()
    sample = args.sent
    sample = str(sample)
    predict(sample)
