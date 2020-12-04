import pdb
import sys
import operator
from collections import OrderedDict
import subprocess
import numpy as  np
import json
import math
from transformers import *
import sys
import urllib.parse
import argparse
import requests



try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')


def gen_line_stats(model_path,do_lower,ref_text,output):
    tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case=do_lower)
    line_stats = {}
    count = 0
    total_len = 0
    total_tokenized_len = 0
    max_t_len = 0
    max_l_len = 0
    with open(ref_text,"r") as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            if (len(line) <= 1):
                continue
            count += 1
            line = line.strip()
            line_len = len(line.split())
            if (do_lower):
                line = line.lower()
            tokenized_text = tokenizer.tokenize(line)
            tokenized_len = len(tokenized_text)
            total_tokenized_len += tokenized_len
            total_len += line_len
            if (tokenized_len > max_t_len):
                print(line_len,tokenized_len)
                max_t_len = tokenized_len
                max_l_len = line_len
            if (line_len not in line_stats):
                line_stats[line_len] = tokenized_len
            else: 
                line_stats[line_len] += tokenized_len
            print("Line {}: {}".format(line_len, tokenized_len))
    sorted_d = OrderedDict(sorted(line_stats.items(), key=lambda kv: kv[1], reverse=True))
    print("Total Stats: Average line len {}.  Average tokenized len {} Average ratio: {} Max tok len {}:{}".format(float(total_len)/count, float(total_tokenized_len)/count, float(total_tokenized_len)/total_len,max_t_len,max_l_len))
    with open(output,"w") as wfp:
        for term in sorted_d:
            wfp.write(str(term) + " " + str(sorted_d[term]) + "\n")
        wfp.close()


DEFAULT_MODEL_PATH="bert-large-uncased"
DEFAULT_TO_LOWER=0

def main():
    parser = argparse.ArgumentParser(description="BERT vocab inverse frequency generator",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', action="store", dest="model", default=DEFAULT_MODEL_PATH,help='BERT pretrained models, or custom model path')
    parser.add_argument('-tolower', action="store", dest="tolower", default=DEFAULT_TO_LOWER,type=int,help='Convert tokens to lowercase. Set to 1')
    parser.add_argument('-ref_text', action="store", dest="ref_text",default="", help='Reference text file with sentences for frequency gen')
    parser.add_argument('-output', action="store", dest="output",default="",help='Output file with frequencies')

    results = parser.parse_args()
    gen_line_stats(results.model,results.tolower,results.ref_text,results.output)



if __name__ == "__main__":
    main()
