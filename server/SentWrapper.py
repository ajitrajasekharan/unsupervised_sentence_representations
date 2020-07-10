import torch
import subprocess
from transformers import *
import pdb
import operator
from collections import OrderedDict
import numpy as np
import json

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


top_k = 3
DESC_FILE="./common_descs.txt"

try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')

def read_descs(file_name):
    ret_dict = {}
    with open(file_name) as fp:
        line = fp.readline().rstrip("\n")
        if (len(line) >= 1):
            ret_dict[line] = 1
        while line:
            line = fp.readline().rstrip("\n")
            if (len(line) >= 1):
                ret_dict[line] = 1
    return ret_dict

class SentWrapper:
    def __init__(self, path):
        self.path = path
        self.tokenizer = BertTokenizer.from_pretrained(path,do_lower_case=False) ### Set this to to True for uncased models
        self.model = BertForMaskedLM.from_pretrained(path)
        self.model.eval()
        self.descs = read_descs(DESC_FILE)
        #pdb.set_trace()


    def punct_sentence(self,text):
        text = text.rstrip(".")
        text = text + " entity"
        text = '[CLS]' + text + '[SEP]'
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Create the segments tensors.
        segments_ids = [0] * len(tokenized_text)

        masked_index = 0
        original_masked_index = 0

        for i in range(len(tokenized_text)):
            if (tokenized_text[i] == "entity"):
                masked_index = i
                original_masked_index = i
                break
        assert (masked_index != 0)
        tokenized_text[masked_index] = "[MASK]"
        indexed_tokens[masked_index] = 103
        print(tokenized_text)
        print(masked_index)
        results_dict = {}

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        ret_dict = {}
        ret_dict["tokenized"] = tokenized_text[1:-2]
        match_arr = []
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
            for word in range(len(tokenized_text)):
                if (word == 0 or word >= len(tokenized_text) -2):
                    continue
                masked_index = word
                for i in range(len(predictions[0][0][masked_index])):
                    tok = self.tokenizer.convert_ids_to_tokens([i])[0]
                    results_dict[tok] = float(predictions[0][0][masked_index][i].tolist())
                k = 0
                sorted_d = OrderedDict(sorted(results_dict.items(), key=lambda kv: kv[1], reverse=True))
                debug_count = 0
                for j in sorted_d:
                    if (j not in self.descs):
                        continue
                    match_arr.append(j)
                    k += 1
                    if (k >= top_k):
                        break
        ret_dict["semantic"] = match_arr
        return json.dumps(ret_dict)



def main():
    MODEL_PATH='bert-large-cased'
    singleton = SentWrapper(MODEL_PATH)
    out = singleton.punct_sentence("He had pina colada and hamburgers for lunch")
    print(out)


if __name__ == '__main__':
    main()

