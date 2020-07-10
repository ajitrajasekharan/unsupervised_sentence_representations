
import pdb
import sys
from collections import OrderedDict
import numpy as  np
import math

def read_terms(terms_file):
    terms_dict =  {}
    count = 0
    with open(terms_file) as fin:
        for line in fin:
            line = line.strip("\n")
            if (len(line) >= 1):
                val = int(line.split()[1])
                if (val in terms_dict):
                    terms_dict[val]  += 1
                else:
                    terms_dict[val]  = 1
                count += 1
    sorted_d = OrderedDict(sorted(terms_dict.items(), key=lambda kv: kv[0], reverse=False))
    #print("count of tokens:",count, len(terms_dict))
    for val in sorted_d:
        print(str(val)+','+str(sorted_d[val]))



if __name__ == "__main__":
    read_terms(sys.argv[1])
