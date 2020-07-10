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


COMP_REF = "ref"
COMP_MIN = "min"
COMP_COS = "cos"


try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')

def read_embeddings(embeds_file):
    with open(embeds_file) as fp:
        embeds_dict = json.loads(fp.read())
    return embeds_dict



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

def read_term_freqs(terms_file):
    terms_dict = OrderedDict()
    with open(terms_file) as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            toks = term.split()
            if (len(toks) == 2):
                terms_dict[toks[0]] = float(1.0/int(toks[1]))
                count += 1
    print("count of term frequency terms:", len(terms_dict))
    return terms_dict

class UnsupSE:
    def __init__(self, model_path,do_lower, terms_file,embeds_file,term_freqs_file,cache_embeds,normalize):
        do_lower = True if do_lower == 1 else False
        self.tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case=do_lower)
        self.terms_dict = read_terms(terms_file)
        self.embeddings = read_embeddings(embeds_file)
        self.term_freqs = read_term_freqs(term_freqs_file)
        self.cache = cache_embeds
        self.embeds_cache = {}
        self.cosine_cache = {}
        self.dist_threshold_cache = {}
        self.normalize = normalize



    def get_embedding(self,text,tokenize=True):
        if (self.cache and text in self.embeds_cache):
            return self.embeds_cache[text]
        if (tokenize):
            tokenized_text = self.tokenizer.tokenize(text)
        else:
            tokenized_text = text.split()
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        #print(text,indexed_tokens)
        vec =  self.get_vector(indexed_tokens)
        if (self.cache):
                self.embeds_cache[text] = vec
        return vec


    def get_vector(self,indexed_tokens):
        vec = None
        if (len(indexed_tokens) == 0):
            return vec
        #pdb.set_trace()
        for i in range(len(indexed_tokens)):
            term_vec = self.embeddings[indexed_tokens[i]]
            if (vec is None):
                vec = np.zeros(len(term_vec))
            vec += term_vec
        sq_sum = 0
        for i in range(len(vec)):
            sq_sum += vec[i]*vec[i]
        sq_sum = math.sqrt(sq_sum)
        for i in range(len(vec)):
            vec[i] = vec[i]/sq_sum
        #sq_sum = 0
        #for i in range(len(vec)):
        #    sq_sum += vec[i]*vec[i]
        return vec

    def calc_inner_prod(self,text1,text2,tokenize):
        if (self.cache and text1 in self.cosine_cache and text2 in self.cosine_cache[text1]):
            return self.cosine_cache[text1][text2]
        vec1 = self.get_embedding(text1,tokenize)
        vec2 = self.get_embedding(text2,tokenize)
        if (vec1 is None or vec2 is None):
            #print("Warning: at least one of the vectors is None for terms",text1,text2)
            return 0
        val = np.inner(vec1,vec2)
        if (self.cache):
            if (text1 not in self.cosine_cache):
                self.cosine_cache[text1] = {}
            self.cosine_cache[text1][text2] = val
        return val


# Global functions

def word_weight(b_embeds,words):
        desc_dict = {}
        for word in words:
            word = word.lower()
            if (word in b_embeds.term_freqs):
                desc_dict[word] = b_embeds.term_freqs[word]
            else:
                desc_dict[word] = UNK_WEIGHT
        return desc_dict

def dispatch_request(url):
    max_retries = 10
    attempts = 0
    while True:
        try:
            r = requests.get(url,timeout=1000)
            if (r.status_code == 200):
                return r
        except:
            print("Request:", url, " failed. Retrying...")
        attempts += 1
        if (attempts >= max_retries):
            print("Request:", url, " failed")
            break

def get_signature(line,server_url):
    disp_sent = server_url + urllib.parse.quote(line)
    ret_data = dispatch_request(disp_sent)
    ret_dict = json.loads(ret_data.text)
    print(ret_dict)
    sense_mag_ratio = int(len(ret_dict['semantic'])/len(ret_dict['tokenized']))
    merge_arr = []
    for i in range(len(ret_dict['tokenized'])):
        merge_arr.append(ret_dict['tokenized'][i])
        for j in range(sense_mag_ratio):
            merge_arr.append(ret_dict['semantic'][i*sense_mag_ratio + j])
    return merge_arr

def gen_batch_mode(b_embeds,batch_input_file,server_url,output):
    print("in gen batch mode")
    count = 0
    wfp = open(output,"w")
    with open(batch_input_file,"r") as fp:
        while True:
            count += 1
            line = fp.readline()
            if not line:
                break
            line = line.strip()
            print("Line {}: {}".format(count, line))
            merge_arr = get_signature(line,server_url)
            wfp.write(' '.join(merge_arr) + "\n")
    wfp.close()


def read_lines(sig_file):
    sig_arr = []
    count = 0
    with open(sig_file,"r") as fp:
        while True:
            count += 1
            line = fp.readline()
            if not line:
                break
            line = line.strip()
            print("Line {}: {}".format(count, line))
            sig_arr.append(line.split())
    return sig_arr

def compute_weights(comp_type,max_val,sent1_count,sent2_count):
    if (comp_type == COMP_MIN):
        return max_val* min(sent1_count, sent2_count)
    elif (comp_type == COMP_REF):
        return max_val * sent1_count
    else:
        return max_val #results will not be good if we choose this option. Stop words will dominate



def compare_pair(b_embeds,sent1_arr,sent2_arr,debug_fp,orig_sent1,orig_sent2,weight_comp):
    tokenize = False
    scores = OrderedDict()
    sent1_counts = word_weight(b_embeds,sent1_arr)
    sent2_counts = word_weight(b_embeds,sent2_arr)
    sent1_len = len(sent1_arr)
    sent2_len = len(sent2_arr)
    selective_pick_count = 0
    full_count = 0
    act_l_count = 0
    act_r_count = 0
    for l_term in sent1_arr:
            max_val = -1
            max_term = ""
            if (len(l_term) == 1):
                continue
            act_l_count += 1
            for r_term in sent2_arr:
                if (len(r_term) == 1):
                    continue
                act_r_count += 1
                full_count += 1
                val = b_embeds.calc_inner_prod(l_term,r_term,tokenize)
                if (val > max_val):
                    max_term = r_term
                    max_val = val
            phrase = l_term + '_' + max_term
            if (len(max_term) == 0):
                continue
            if (phrase in scores):
                # Two weightings are done
                #Weighting 1
                scores[phrase] += compute_weights(weight_comp,max_val,sent1_counts[l_term.lower()],sent2_counts[max_term.lower()])
                #Weighting 2
                if (max_val < .9):
                    scores[phrase] += max_val * sent1_counts[l_term.lower()]
                    selective_pick_count += 1
            else:
                scores[phrase] = compute_weights(weight_comp,max_val,sent1_counts[l_term.lower()],sent2_counts[max_term.lower()])
                if (max_val < .9):
                    scores[phrase] = max_val * sent1_counts[l_term.lower()]
                    selective_pick_count += 1
    total_score = 0
    for i in scores:
        total_score += scores[i]
    ret_val = total_score/(full_count + selective_pick_count)
    act_r_count /= act_l_count #r_count is counted l_count_times. So scaling it down to true r_count_value
    len_scale = (float(act_r_count)/act_l_count) if (act_r_count < act_l_count) else 1
    ret_val *= len_scale
    debug_fp.write(orig_sent1 + "\n")
    debug_fp.write(orig_sent2 + "\n")
    debug_fp.write("Total score:" + str(total_score) + "  count: " + str(len(sent1_arr))  + "\n")
    debug_fp.write(' '.join(sent1_arr) + "\n")
    debug_fp.write(' '.join(sent2_arr) + "\n")
    debug_fp.write(str(sent1_counts) + "\n")
    debug_fp.write(str(sent2_counts) + "\n")
    sorted_d = OrderedDict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
    debug_fp.write(str(sorted_d) + "\n")
    debug_fp.write("*** Score: " + str(ret_val) + "\n\n\n\n")
    return  ret_val,sorted_d


def compute_final_score(c_score,pos_index):
    return  min(((1.0/pos_index) + c_score),1)

def compare_sentences(b_embeds,sent,sent_sig_arr,ref_text_arr,ref_sig_arr,weight_comp):
    ref_scores = {}
    descs_match_dict = {}
    final_scores_dict = OrderedDict()
    debug_fp = open("debug.txt","w")
    for i in range(len(ref_text_arr)):
        ret_score,descs_dict = compare_pair(b_embeds,sent_sig_arr,ref_sig_arr[i],debug_fp,sent,' '.join(ref_text_arr[i]),weight_comp)
        key_val = '_'.join(ref_text_arr[i])
        ref_scores[key_val] = ret_score
        descs_match_dict[key_val] = descs_dict
    sorted_d = OrderedDict(sorted(ref_scores.items(), key=lambda kv: kv[1], reverse=True))
    first = True
    max_score = 0
    pos_index = 1
    for sent in sorted_d:
        if (first):
            max_score = sorted_d[sent]
            first = False
        final_score = compute_final_score(sorted_d[sent]/max_score,pos_index)
        final_scores_dict[sent] = final_score
        debug_fp.write("Score for sent: " + sent.replace('_',' ') + "      :" +  str(final_score) + "\n\n")
        pos_index += 1
    debug_fp.close()
    return final_scores_dict,descs_match_dict

def get_descs(sent,descs_match_dict,match_limit):
    ret_arr = []
    match_dict = descs_match_dict[sent]
    count = 0
    for match in match_dict:
        ret_arr.append(match)
        count += 1
        if (count == match_limit):
            break
    return ' '.join(ret_arr)


def output_scores(scores_dict,results_limit,descs_match_dict):
    count = 0
    max_score = 0
    descs_limit = DESCS_LIMIT
    print()
    for sent in scores_dict:
        descs = get_descs(sent,descs_match_dict,descs_limit)
        print("{}] {} : {} ({})\n".format(count+1,sent.replace('_',' '),scores_dict[sent],descs))
        count += 1
        if (count >= results_limit):
            break


def test_interactive_mode(b_embeds,ref_text_file,ref_sig_file,server_url,limit,weight_comp):
    ref_sig_arr = read_lines(ref_sig_file)
    ref_text_arr = read_lines(ref_text_file)
    assert(len(ref_sig_arr) == len(ref_text_arr))
    print("Match mode is :",weight_comp)
    while True:
        print("Enter sentence")
        sent = input()
        sent_sig_arr = get_signature(sent,server_url)
        scores_dict,descs_match_dict = compare_sentences(b_embeds,sent,sent_sig_arr,ref_text_arr,ref_sig_arr,weight_comp)
        output_scores(scores_dict,limit,descs_match_dict)
    print("in test interactive mode")
    pass


def read_batch_input(batch_file):
    line_arr = []
    with open(batch_file) as fin:
        count = 1
        for line in fin:
            line = line.strip("\n")
            if (len(line) >= 1):
                line_arr.append(line)
    print("Read {} line".format(len(line_arr)))
    return line_arr

def get_sent_index(ref_arr,sent):
    j = 0
    for ref_sent in ref_arr:
        if (sent == ' '.join(ref_sent)):
            return j
        j += 1
    assert(0)

def test_batch_mode(b_embeds,ref_text_file,ref_sig_file,server_url,batch_test_input_file,output_file,weight_comp):
    print("in test batch mode")
    ref_sig_arr = read_lines(ref_sig_file)
    ref_text_arr = read_lines(ref_text_file)
    assert(len(ref_sig_arr) == len(ref_text_arr))
    batch_input_arr = read_batch_input(batch_test_input_file)
    similarity_matrix = np.zeros(len(batch_input_arr)*len(ref_sig_arr)).reshape(len(batch_input_arr),len(ref_sig_arr))
    d_file = "MATRIX_" + '_'.join(output_file.split('.')[0].split('/')) + ".txt"
    dfp = open(d_file,"w")
    for i in range(len(batch_input_arr)):
            sent = batch_input_arr[i]
            #sent_sig_arr = get_signature(sent,server_url)
            sent_sig_arr = ref_sig_arr[i]
            scores_dict,descs_match_dict = compare_sentences(b_embeds,sent,sent_sig_arr,ref_text_arr,ref_sig_arr,weight_comp)
            assert(len(scores_dict) == len(ref_sig_arr))
            for ref_sent in scores_dict:
                sent_index = get_sent_index(ref_text_arr,ref_sent.replace('_',' '))
                score = scores_dict[ref_sent]
                if (similarity_matrix[i][sent_index] != 0):
                    pdb.set_trace()
                #assert(similarity_matrix[i][sent_index] ==  0)
                #assert(similarity_matrix[sent_index][i] == 0)
                similarity_matrix[i][sent_index] = score
                #similarity_matrix[sent_index][i] = score
                dfp.write(str(i) +"," +str(sent_index)+"] " +sent + "  |  " +ref_sent.replace('_',' ') +" : " +str(score) + "\n")
                print("Writing " + str(i) + " " + str(sent_index) + " " + str(score))
    dfp.close()
    with open(output_file,"wb") as wfp:
        np.save(wfp,similarity_matrix)
    fp = open(output_file,"rb")
    arr = np.load(fp)






DEFAULT_MODEL_PATH="bert-large-cased"
DEFAULT_TO_LOWER=0
DEFAULT_TOKENIZE=0
DEFAULT_VOCAB_FILE="vocab.txt"
DEFAULT_VECTOR_FILE="bert_vectors.txt"
DEFAULT_RESULTS_LIMIT=20
DEFAULT_SERVER_URL="http://127.0.0.1:8900/dummy/"
DEFAULT_TERM_FREQS="term_freq.txt"
DEFAULT_WEIGHT_COMP="min"
UNK_WEIGHT = .5
DESCS_LIMIT = 50

def main():
    parser = argparse.ArgumentParser(description="Sentence representation generator and tester",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', action="store", dest="model", default=DEFAULT_MODEL_PATH,help='BERT pretrained models, or custom model path')
    parser.add_argument('-tolower', action="store", dest="tolower", default=DEFAULT_TO_LOWER,type=int,help='Convert tokens to lowercase. Set to 1 only for uncased models')
    parser.add_argument('-vocab', action="store", dest="vocab", default=DEFAULT_VOCAB_FILE,help='vocab file')
    parser.add_argument('-vectors', action="store", dest="vectors", default=DEFAULT_VECTOR_FILE,help='bert vector file')
    parser.add_argument('-mode', action="store", dest="mode", help='gen/interactive/test_batch')
    parser.add_argument('-ref_text', action="store", dest="ref_text",default="", help='Reference text file with sentences for testing/gen')
    parser.add_argument('-ref_sig', action="store", dest="ref_sig",default="", help='Reference signature file with sentences for testing')
    parser.add_argument('-test_batch', action="store", dest="test_batch",default="",help='Batch file with sentences for testing')
    parser.add_argument('-output', action="store", dest="output",default="",help='Output file for generation as well as test batch output')
    parser.add_argument('-limit', action="store", dest="limit", default=DEFAULT_RESULTS_LIMIT,type=int,help='Number of results to display')
    parser.add_argument('-server_url', action="store", dest="server_url", default=DEFAULT_SERVER_URL,help='URL of mask server to connect to')
    parser.add_argument('-term_freqs', action="store", dest="term_freqs", default=DEFAULT_TERM_FREQS,help='token frequencies in a corpus')
    parser.add_argument('-weight_comp', action="store", dest="weight_comp", default=DEFAULT_WEIGHT_COMP, help='Type of weighting of cosine values of word pairs. min,ref,cos.  min -  Use the min of both words. ref- use the frequency count of input sentence word only.  cos - just use cosine weights alone')

    results = parser.parse_args()
    server_url = results.server_url
    limit = results.limit
    b_embeds = UnsupSE(results.model,results.tolower,results.vocab,results.vectors,results.term_freqs,True,True) #True - for cache embeds; normalize - True
    if (results.mode == "gen"):
        if (len(results.ref_text) == 0 or len(results.output) == 0):
            print("Reference text file needs to be specified")
        else:
            gen_batch_mode(b_embeds,results.ref_text,server_url,results.output)
    elif (results.mode == "interactive"):
            if (len(results.ref_text) == 0 or len(results.ref_sig) == 0):
                print("Reference text and sig file needs to be specified for interactive testing")
            else:
                test_interactive_mode(b_embeds,results.ref_text,results.ref_sig,server_url,limit,results.weight_comp)
    elif (results.mode == "test_batch"):
            if (len(results.ref_text) == 0 or len(results.ref_sig) == 0 or len(results.test_batch) == 0 or len(results.output) == 0):
                print("Reference text,sig,batch input text file and output file needs to be specified for batch testing")
            else:
                test_batch_mode(b_embeds,results.ref_text,results.ref_sig,server_url,results.test_batch,results.output,results.weight_comp)
    else:
       parser.print_help()



if __name__ == "__main__":
    main()
