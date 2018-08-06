# POS-Tagging-using-Neural-Network-Models
Contains implementation of models like BiLSTM CRF, Hierarchical BiLSTM for POS Tagging.

Data in the pickle file is expected to have the following format:

[(Sentence1,Tags_of_Sentence1),(Sentence2,Tags_of_Sentence2),(Sentence3,Tags_of_Sentence3)]

Example from the training pickle file "total_data.pickle":

[

( [ 'word1', 'word2', 'word3', 'word4' ,'.'], ['N-NNP', 'N-NNP', 'N_NN', 'V_VM_VNF' 'RD_PUNC']),

([ 'word1', 'word2', 'word3', 'word4' ,'word5','.'], ['PR_PRP', 'N_NN', 'N_NN', 'PSP', 'V_VM_VF', 'RD_PUNC'])

]
