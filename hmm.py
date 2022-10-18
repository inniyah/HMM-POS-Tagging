from collections import defaultdict

import numpy as np

from utils import get_word_tag, assign_unk, processing

corpus_path = "WSJ_02-21.pos"
def training_data(corpus_path):
    
    with open(corpus_path, 'r') as f:
        training_corpus = f.readlines()
    return training_corpus

def create_dictionaries(training_corpus, vocab2idx):
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)

    prev_tag = '--s--'

    for tok_tag in training_corpus:

        tok, tag = get_word_tag(tok_tag, vocab2idx)
        transition_counts[(prev_tag, tag)] += 1
        emission_counts[(tag, tok)] += 1
        tag_counts[tag] += 1
        prev_tag = tag

    return emission_counts, transition_counts, tag_counts

 
def create_transition_matrix(transition_counts, tag_counts, alpha):
    all_tags = sorted(tag_counts.keys())
    num_tags = len(all_tags)

    # initialize the transition matrix 'A'
    A = np.zeros((num_tags, num_tags))

    # get the unique transition tuples (prev POS, cur POS)
    trans_keys = set(transition_counts.keys())

    for i in range(num_tags):
        for j in range(num_tags):
            # initialize the count of (prev POS, cur POS)
            count = 0

            key = (all_tags[i], all_tags[j])
            if key in transition_counts:
                count = transition_counts[key]
            count_prev_tag = tag_counts[all_tags[i]]

            A[i, j] = (count + alpha) / (count_prev_tag + alpha * num_tags)

    return A

def create_emission_matrix(emission_counts, tag_counts, vocab2idx, alpha):
    num_tags = len(tag_counts)
    all_tags = sorted(tag_counts.keys())
    num_words = len(vocab2idx)

    B = np.zeros((num_tags, num_words))
    emis_keys = set(list(emission_counts.keys()))
    for i in range(num_tags):
        for j in range(num_words):
            count = 0

            key =  (all_tags[i], vocab2idx[j])
            if key in emission_counts:
                count = emission_counts[key]
            count_tag = tag_counts[all_tags[i]]

            B[i, j] = (count + alpha) / (count_tag + alpha * num_words)
    return B

def initialize(A, B, tag_counts, vocab2idx, states, prep_tokens):
    num_tags = len(tag_counts)
    best_probs = np.zeros((num_tags, num_tags))
    best_paths = np.zeros((num_tags, len(prep_tokens)), dtype=int)
    s_idx = states.index('--s--')

    for i in range(num_tags):
        if A[s_idx, i] == 0:
            best_probs[i, 0] = float('-inf')
        else:
            best_probs[i,0] = np.log(A[s_idx, i]) + np.log(B[i, vocab2idx[prep_tokens[0]]])

    return best_probs, best_paths

def viterbi_forward(A, B, prep_tokens, best_probs, best_paths, vocab2idx):
    num_tags = best_probs.shape[0]
    for i in range(1, len(prep_tokens)):
        for j in range(num_tags):
            best_prob_i = float('-inf')
            best_path_i = None

            for k in range(num_tags):
                prob = best_probs[k,i-1]+np.log(A[k,j]) +np.log(B[j,vocab2idx[prep_tokens[i]]])
                if prob > best_prob_i:
                    best_prob_i = prob
                    best_path_i = k
            best_probs[j, i] = best_prob_i
            best_paths[j, i] = best_path_i
    return best_probs, best_paths


def viterbi_backward(best_probs, best_paths, states):
    m = best_paths.shape[1]
    z = [None] * m
    num_tags = best_probs.shape[0]

    best_prob_for_last_word = float('-inf')
    pred = [None] * m

    for k in range(num_tags):
        if best_probs[k, m - 1] > best_prob_for_last_word:
            best_prob_for_last_word = best_probs[k, m - 1]
            z[m - 1] = k
    pred[m - 1] = states[z[m - 1]]

    for i in range(m-1, -1, -1):
        pos_tag_for_word_i = z[i]
        z[i - 1] = best_paths[pos_tag_for_word_i,i]
        pred[i - 1] = states[z[i - 1]]
    return pred
