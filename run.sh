#python sentence_representations.py -model bert-large-cased -tolower 0  -vocab vocab.txt -vectors bert_vectors.txt -mode gen -ref_text data/big_sents.txt -output sigs/big_sig.txt
#python sentence_representations.py -model bert-large-cased -tolower 0  -vocab vocab.txt -vectors bert_vectors.txt -mode interactive -ref_text data/big_sents_with_labels.txt -ref_sig sigs/big_sig.txt 
#python sentence_representations.py -model bert-large-cased -tolower 0  -vocab vocab.txt -vectors bert_vectors.txt -mode test_batch -ref_text data/big_sents.txt -ref_sig sigs/big_sig.txt -output heatmaps/big_sents.npy -test_batch data/big_sents.txt

#python sentence_representations.py -model bert-large-cased -tolower 0  -vocab vocab.txt -vectors bert_vectors.txt -mode gen -ref_text data/small_sents.txt -output sigs/small_sig.txt
python sentence_representations.py -model bert-large-cased -tolower 0  -vocab vocab.txt -vectors bert_vectors.txt -mode interactive -ref_text data/small_sents.txt -ref_sig sigs/small_sig.txt 
#python sentence_representations.py -model bert-large-cased -tolower 0  -vocab vocab.txt -vectors bert_vectors.txt -mode test_batch -ref_text data/small_sents.txt -ref_sig sigs/small_sig.txt -output heatmaps/small_sents.npy -test_batch data/small_sents.txt
