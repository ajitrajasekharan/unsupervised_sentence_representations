import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import pdb
import sys

# Some texts of different lengths.

def load_sentences(filename):
	sentences = []
	with open(filename) as fp:
		for line in fp:
			print(line.rstrip())
			sentences.append(line.rstrip())	
	return sentences


def main(filename):
	english_sentences = load_sentences(filename)
	embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

	# Compute embeddings.
	en_result = embed(english_sentences)

# Compute similarity matrix. Higher score indicates greater similarity.
	similarity_matrix = np.inner(en_result, en_result)
	output_file = "USE_" + filename.split('.')[0] + ".npy"
	with open(output_file,"wb") as fp:
		np.save(fp,similarity_matrix)
	print(similarity_matrix)


if __name__ == "__main__":
	if (len(sys.argv) != 2):
		print("Specify file name with sentences")
	else:
		main(sys.argv[1])
