#!/usr/bin/env python3

import argparse
import io
import json
import logging
import math
import os
import pickle
import signal
import sys
import time
import traceback

from pprint import pprint
from datetime import datetime

from numpy import save
from numpy import load

MY_PATH = os.path.normpath(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(MY_PATH, '.')))

from hmm import build_vocab2idx, create_dictionaries, create_transition_matrix, create_emission_matrix, initialize, viterbi_forward, viterbi_backward, training_data
from utils import get_word_tag, assign_unk, processing
from build_vocabulary import build_vocab
from syntok.tokenizer import Tokenizer

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

exit_program = False

def exit():
    global exit_program
    exit_program = True

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

CORPUS_PATH = "data/WSJ_02-21.pos"
ALPHA = 0.001

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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def run_load(args):
    os.makedirs(os.path.join(MY_PATH, 'tmp'), exist_ok=True)
    logging.info("Building vocabulary index")
    vocab2idx = build_vocab2idx(CORPUS_PATH)
    logging.info("Saving vocabulary index to 'vocab.pkl'")
    with open(os.path.join(MY_PATH, 'tmp', 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab2idx, f)
    logging.info(f"Training the system with the data corpus in '{CORPUS_PATH}'")
    training_corpus = training_data(CORPUS_PATH)
    logging.info("Creating dictionaries")
    emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab2idx)
    states = sorted(tag_counts.keys())
    alpha = ALPHA
    logging.info("Creating transition matrix")
    A = create_transition_matrix(transition_counts, tag_counts, alpha)
    logging.info("Creating emission matrix")
    B = create_emission_matrix(emission_counts, tag_counts, list(vocab2idx), alpha)
    logging.info("Saving transition matrix and emission matrix")
    save(os.path.join(MY_PATH, 'tmp', 'A.npy'), A)
    save(os.path.join(MY_PATH, 'tmp', 'B.npy'), B)
    return 0

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def run_predict(args):
    sample = 'My heart is always breaking for the ghosts that haunt this room.'
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

LOG_SIMPLE_FORMAT = "[%(pathname)s:%(lineno)d] '%(message)s'"
LOG_CONSOLE_FORMAT = "[%(pathname)s:%(lineno)d] [%(asctime)s]: '%(message)s'"
LOG_FILE_FORMAT = "[%(levelname)s] [%(pathname)s:%(lineno)d] [%(asctime)s] [%(name)s]: '%(message)s'"

LOGS_DIRECTORY = None

class ColorStderr(logging.StreamHandler):
    def __init__(self, fmt=None):
        class AddColor(logging.Formatter):
            def __init__(self):
                super().__init__(fmt)
            def format(self, record: logging.LogRecord):
                msg = super().format(record)
                # Green/Cyan/Yellow/Red/Redder based on log level:
                color = '\033[1;' + ('32m', '36m', '33m', '31m', '41m')[min(4,int(4 * record.levelno / logging.FATAL))]
                return color + record.levelname + '\033[1;0m: ' + msg
        super().__init__(sys.stderr)
        self.setFormatter(AddColor())

def load_config(cfg_filename='config.json'):
    try:
        with open(os.path.join(MY_PATH, cfg_filename), 'r') as cfg:
            config = json.loads(cfg.read())
            for k in config.keys():
                if k in globals():
                    globals()[k] = config[k]
    except FileNotFoundError:
        pass

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--quiet", help="set logging to ERROR",
                        action="store_const", dest="loglevel",
                        const=logging.ERROR, default=logging.INFO)
    parser.add_argument("-d", "--debug", help="set logging to DEBUG",
                        action="store_const", dest="loglevel",
                        const=logging.DEBUG, default=logging.INFO)

    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument("--load", help="load data",
                        action="store_const", dest="command",
                        const="load", default=None)
    command_group.add_argument("--predict", help="run prediction",
                        action="store_const", dest="command",
                        const="predict", default=None)

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    log_console_handler = ColorStderr(LOG_CONSOLE_FORMAT)
    log_console_handler.setLevel(args.loglevel)
    logger.addHandler(log_console_handler)

    if not LOGS_DIRECTORY is None:
        now = datetime.now()
        logs_dir = os.path.abspath(os.path.join(MY_PATH, LOGS_DIRECTORY, f"{now.strftime('%Y%m%d')}"))
        os.makedirs(logs_dir, exist_ok=True)
        log_filename = f"{now.strftime('%Y%m%d')}_{now.strftime('%H%M%S')}.txt"
        log_file_handler = logging.FileHandler(os.path.join(logs_dir, log_filename))
        log_formatter = logging.Formatter(LOG_FILE_FORMAT)
        log_file_handler.setFormatter(log_formatter)
        log_file_handler.setLevel(logging.DEBUG)
        logger.addHandler(log_file_handler)
        logging.info(f"Storing log into '{log_filename}' in '{logs_dir}'")

    ret = 0
    try:
        if args.command == 'load':
            ret = run_load(args)
        if args.command == 'predict':
            ret = run_predict(args)

    except Exception as e:
        logging.error(f"{type(e).__name__}: {e}")
        logging.error(traceback.format_exc())
        #~ logging.error(sys.exc_info()[2])
        ret = -1

    return ret

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    #~ faulthandler.enable()

    #~ def sigint_handler(signum, frame):
    #~     global exit_program
    #~     logging.warning("CTRL-C was pressed")
    #~     exit_program = True
    #~     sys.exit(-2)
    #~ signal.signal(signal.SIGINT, sigint_handler)

    #~ load_config()
    sys.exit(main())
