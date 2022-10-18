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

TAGS= {
    'CC': ( 'conjunction, coordinating', 'and, or, but' ),
    'CD': ( 'cardinal number', 'five, three, 13%' ),
    'DT': ( 'determiner', 'the, a, these' ),
    'EX': ( 'existential there', 'there were six boys' ),
    'FW': ( 'foreign word', 'mais' ),
    'IN': ( 'conjunction, subordinating or preposition', 'of, on, before, unless' ),
    'JJ': ( 'adjective', 'nice, easy' ),
    'JJR': ( 'adjective, comparative', 'nicer, easier' ),
    'JJS': ( 'adjective, superlative', 'nicest, easiest' ),
    'LS': ( 'list item marker', ' ' ),
    'MD': ( 'verb, modal auxillary', 'may, should' ),
    'NN': ( 'noun, singular or mass', 'tiger, chair, laughter' ),
    'NNS': ( 'noun, plural', 'tigers, chairs, insects' ),
    'NNP': ( 'noun, proper singular', 'Germany, God, Alice' ),
    'NNPS': ( 'noun, proper plural', 'we met two Christmases ago' ),
    'PDT': ( 'predeterminer', 'both his children' ),
    'POS': ( 'possessive ending', '\'s' ),
    'PRP': ( 'pronoun, personal', 'me, you, it' ),
    'PRP$': ( 'pronoun, possessive', 'my, your, our' ),
    'RB': ( 'adverb', 'extremely, loudly, hard ' ),
    'RBR': ( 'adverb, comparative', 'better' ),
    'RBS': ( 'adverb, superlative', 'best' ),
    'RP': ( 'adverb, particle', 'about, off, up' ),
    'SYM': ( 'symbol', '%' ),
    'TO': ( 'infinitival to', 'what to do?' ),
    'UH': ( 'interjection', 'oh, oops, gosh' ),
    'VB': ( 'verb, base form', 'think' ),
    'VBZ': ( 'verb, 3rd person singular present', 'she thinks' ),
    'VBP': ( 'verb, non-3rd person singular present', 'I think' ),
    'VBD': ( 'verb, past tense', 'they thought' ),
    'VBN': ( 'verb, past participle', 'a sunken ship' ),
    'VBG': ( 'verb, gerund or present participle', 'thinking is fun' ),
    'WDT': ( 'wh-determiner', 'which, whatever, whichever' ),
    'WP': ( 'wh-pronoun, personal', 'what, who, whom' ),
    'WP$': ( 'wh-pronoun, possessive', 'whose, whosever' ),
    'WRB': ( 'wh-adverb', 'where, when' ),
    '.': ( 'punctuation mark, sentence closer', '.;?*' ),
    ',': ( 'punctuation mark, comma', ',' ),
    ':': ( 'punctuation mark, colon', ':' ),
    '(': ( 'contextual separator, left paren', '(' ),
    ')': ( 'contextual separator, right paren', ')' ),
}

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
    for tok, tag in res:
        print(f"{tok}\t{tag}\t{TAGS[tag][0]} ({TAGS[tag][1]})")

if __name__ == "__main__":
    args = parse_argument()
    sample = args.sent
    sample = str(sample)
    predict(sample)
