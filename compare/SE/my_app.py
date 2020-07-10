import sys
import pdb
import numpy as np
"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
from sentence_transformers import SentenceTransformer
import scipy.spatial

def init_model(model_path):
	#embedder = SentenceTransformer('bert-base-nli-mean-tokens')
	print("Loading model:",model_path)
	embedder = SentenceTransformer(model_path)
	return embedder


TYPE_REG="REGULAR"
TYPE_FT="FT"
g_curr_type = TYPE_REG
def specific_process(line,curr_type):
	ret_line = ""
	if (g_curr_type == "FT"):
		ret_line =  ' '.join(line.split(',')[2:]).strip('\n')
	else:
		ret_line =  line.strip('\n')
	return ret_line


def gen_vectors(file_name,model_name,embedder,saved):
	orig_sents = []
	sents = []
	g_curr_type = TYPE_FT
	#curr_type = TYPE_REG

	#with open(file_name,"r",encoding='ISO-8859-1') as fp:
	#	for line in fp:
	#		sents.append(line)
	with open(file_name,"r") as fp:
		for line in fp:
			sents.append(specific_process(line,g_curr_type))
			orig_sents.append(line.strip('\n'))
	if (saved):
		print("Loading vectors:")
		vecs = np.load(model_name)
		print("Loaded vectors:",len(vecs))
	else:
		print("Generating vectors:")
		vecs = embedder.encode(sents)
		print("Generated vectors:",len(vecs))
	return vecs,orig_sents


def test_model(queries,embedder,corpus_embeddings,sents):
	query_embeddings = embedder.encode(queries)
	# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
	closest_n = 20
	wfp = open("debug.txt","w")
	for query, query_embedding in zip(queries, query_embeddings):
		distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
		results = zip(range(len(distances)), distances)
		results = sorted(results, key=lambda x: x[1])

		print("\n\n======================\n\n")
		print("Query:", query)
		wfp.write("Query:" +  query + "\n\n")
		print("\nTop " + str(closest_n) + "  most similar sentences in corpus:")

		count = 1
		for idx, distance in results[0:closest_n]:
			print(count,"]",sents[idx].strip(), "(Score: %.4f)" % (1-distance))
			wfp.write(str(count) + "]" + sents[idx].strip() +  ("(Score: %.4f)" % (1-distance)) + "\n\n")
			count += 1
	wfp.close()

def interactive_test_model(embedder,corpus_embeddings,sents):
	while True:
		print("Enter input sentence:")
		sent = input()
		test_model([sent],embedder,corpus_embeddings,sents)



def gen_inner(corpus_embeddings,sents):
	assert(len(corpus_embeddings) == len(sents))
	s_count = len(sents)	
	similarity_matrix = np.zeros(s_count*s_count).reshape(s_count,s_count)
	i = 0
	for query, query_embedding in zip(sents, corpus_embeddings):
		distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
		results = zip(range(len(distances)), distances)
		for j, distance in results:
			score = 1 - distance
			assert(similarity_matrix[i][j] == 0)
			similarity_matrix[i][j] = score
		i += 1			
	return(similarity_matrix)

def gen_matrix(file_name,embedder,corpus_embeddings,sents):
	similarity_matrix = gen_inner(corpus_embeddings,sents)
	output_file = "SE_" + file_name.split('.')[0] + ".npy"
	with open(output_file,"wb") as fp:
		np.save(fp,similarity_matrix)
	print(similarity_matrix)


		

def train_model(file_name,model_type,model_name,saved):
	embedder = init_model(model_type)
	corpus_embeddings,sents = gen_vectors(file_name,model_name,embedder,saved)
	print("Saving vectors:",len(corpus_embeddings))
	np.save(model_name,corpus_embeddings)

def load_and_test_model(file_name,model_type,model_name,saved):
	embedder = init_model(model_type)
	corpus_embeddings,sents = gen_vectors(file_name,model_name,embedder,saved)
	gen_matrix(file_name,embedder,corpus_embeddings,sents)
	interactive_test_model(embedder,corpus_embeddings,sents)



if __name__ == "__main__":
	if (len(sys.argv) != 7):
		print("<input data file> <model path> <model name to train/load> 0/1[saved/gen] 0/1(interactive test) <REGULAR/FT regular data/fasttext data>")
	else:
		saved = int(sys.argv[4])
		test = int(sys.argv[5])
		g_curr_type = sys.argv[6]
		if (test == 0):
			train_model(sys.argv[1],sys.argv[2],sys.argv[3],saved)
		else:
			load_and_test_model(sys.argv[1],sys.argv[2],sys.argv[3],saved)



