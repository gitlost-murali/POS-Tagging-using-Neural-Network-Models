# POS-Tagging-using-Neural-Network-Models
Contains implementation of models like BiLSTM CRF, Hierarchical BiLSTM for POS Tagging.

See "BiLSTM CRF explanation.md" for the explanation of [BiLSTM CRF](https://github.com/Murali81/POS-Tagging-using-Neural-Network-Models/blob/master/BiLSTM%20CRF%20explanation.md).

Data in the "total_data.pickle" pickle file is expected to have the following format:

[(Sentence1,Tags_of_Sentence1),(Sentence2,Tags_of_Sentence2),(Sentence3,Tags_of_Sentence3)]

Example from the training pickle file "total_data.pickle":

[

( [ 'word1', 'word2', 'word3', 'word4' ,'.'], ['N-NNP', 'N-NNP', 'N_NN', 'V_VM_VNF' 'RD_PUNC']),

([ 'word1', 'word2', 'word3', 'word4' ,'word5','.'], ['PR_PRP', 'N_NN', 'N_NN', 'PSP', 'V_VM_VF', 'RD_PUNC'])

]

For Shallow Parser ,

Data for "total_poschunk_data.pickle" must be in the following format,
[

( [ 'word1', 'word2', 'word3', 'word4' ,'.'], [('B-NP','N-NNP'), ('I-NP',N-NNP'), ('O-NP','N_NN'), ('B-VM','V_VM_VNF') ('B-PUNC','RD_PUNC')]),

([ 'word1', 'word2', 'word3', 'word4' ,'word5','.'], [('B-NP','PR_PRP'), ('I-NP','N_NN'), ('I-NP','N_NN'), ('O-NP','PSP'), ('B-VM','V_VM_VF'), ('B-PUNC','RD_PUNC')])

]
