import pdb
import sys
import operator
from collections import OrderedDict
import subprocess
import numpy as  np
import json
import math
from pytorch_transformers import *
import sys
import urllib.parse
import argparse
import requests



try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')

def read_terms(terms_file):
    terms_dict = OrderedDict()
    with open(terms_file) as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            if (len(term) >= 1):
                terms_dict[term] = count
                count += 1
    print("count of tokens:", len(terms_dict))
    return terms_dict

def gen_inverse_frequency(model_path,do_lower,ref_text,output,vocab):
    tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case=do_lower)
    vocab_dict = read_terms(vocab)
    tokenized_dict = {}
    count = 0
    with open(ref_text,"r") as fp:
        while True:
            count += 1
            line = fp.readline()
            if not line:
                break
            line = line.strip()
            tokenized_text = tokenizer.tokenize(line)
            for term in tokenized_text:
                term  = term.lower()
                if (term not in tokenized_dict):
                    tokenized_dict[term]  = 1
                else:
                    tokenized_dict[term] += 1
            print("Line {}: {}".format(count, line))
    sorted_d = OrderedDict(sorted(tokenized_dict.items(), key=lambda kv: kv[1], reverse=True))
    with open(output,"w") as wfp:
        for term in sorted_d:
            wfp.write(term + " " + str(sorted_d[term]) + "\n")
        wfp.close()


DEFAULT_MODEL_PATH="bert-large-cased"
DEFAULT_TO_LOWER=1
DEFAULT_VOCAB_FILE="vocab.txt"

def main():
    parser = argparse.ArgumentParser(description="BERT vocab inverse frequency generator",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', action="store", dest="model", default=DEFAULT_MODEL_PATH,help='BERT pretrained models, or custom model path')
    parser.add_argument('-tolower', action="store", dest="tolower", default=DEFAULT_TO_LOWER,type=int,help='Convert tokens to lowercase. Set to 1')
    parser.add_argument('-vocab', action="store", dest="vocab", default=DEFAULT_VOCAB_FILE,help='vocab file. Used for coverage stats')
    parser.add_argument('-ref_text', action="store", dest="ref_text",default="", help='Reference text file with sentences for frequency gen')
    parser.add_argument('-output', action="store", dest="output",default="",help='Output file with frequencies')

    results = parser.parse_args()
    gen_inverse_frequency(results.model,results.tolower,results.ref_text,results.output,results.vocab)



if __name__ == "__main__":
    main()
