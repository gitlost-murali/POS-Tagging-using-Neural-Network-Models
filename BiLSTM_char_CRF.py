
# coding: utf-8

# In[1]:



import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)


# In[2]:

device = torch.device("cuda:0")


# In[3]:

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score +         torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# In[4]:

def get_char_sequence(word,char_to_idx):
    return [char_to_idx.get(i,0) for i in word]

def generate_sequence(seq,word_to_idx,char_to_idx):
#     print(word_to_idx[seq[0]],get_char_sequence(seq[0],char_to_idx))
    return [[word_to_idx.get(i,0),get_char_sequence(i,char_to_idx)] for i in seq]

def prepare_target(tags, pos_map):
    return autograd.Variable(torch.LongTensor([pos_map.get(i,0) for i in tags]))


# In[ ]:

word_to_ix,char_to_ix={},{}    
tag_to_ix={}

def create_word_char_pos_dicts(totaldata):
    
    tot_data=totaldata[:]    

    word_to_ix['_unk_']=0
    char_to_ix['_unk_']=0
    tag_to_ix['unk']=0
    for sent,tag in tot_data:
        for j in sent:
            if j not in word_to_ix.keys():
                word_to_ix[j] = len(word_to_ix)

            for k in j:
                if k not in char_to_ix.keys():
                    char_to_ix[k] = len(char_to_ix)

        for tg in tag:
            if tg not in tag_to_ix.keys():
                tag_to_ix[tg] = len(tag_to_ix)


# In[5]:

# Make up some training data
START_TAG = "<START>"
STOP_TAG = "<STOP>"

create_word_char_pos_dicts(total_data)

tag_to_ix[START_TAG]=len(tag_to_ix)
tag_to_ix[STOP_TAG]=len(tag_to_ix)


# In[ ]:




# In[6]:

from gensim.models.keyedvectors import KeyedVectors
vectors_file='D:\MyPS3\IIIT H\Pre-Training LM Keras\Large Telugu corpora by ganesh sir\corpus.v3.bin'
word_vectors = KeyedVectors.load_word2vec_format(vectors_file, binary=True, unicode_errors='ignore')


# In[7]:

import numpy as np


# In[8]:

rev_word_idx=dict((v,k) for (k,v) in word_to_ix.items())


# In[9]:

rev_pos_map=dict((v,k) for (k,v) in tag_to_ix.items())


# In[10]:

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, dist_char_size,tag_to_ix, embedding_dim, hidden_dim,char_embedding_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.char_embedding_dim=char_embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        self.char_embeddings=nn.Embedding(dist_char_size,self.char_embedding_dim)
        self.char_lstm = nn.LSTM(self.char_embedding_dim,self.char_embedding_dim,bidirectional=True)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim+(self.char_embedding_dim*2), hidden_dim,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim*2, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim,device=device),
                torch.randn(2, 1, self.hidden_dim,device=device))


    def dyn_init_hidden(self,dimension):
        return (torch.zeros(2,1,dimension,device=device),torch.zeros(2,1,dimension,device=device))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var.cuda() + trans_score.cuda() + emit_score.cuda()
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        
        only_words = []
        char_lstm_outputs =[]
        # [[0, [0, 1, 2]], [1, [3, 4, 5]], [2, [6, 0, 2]], [0, [0, 1, 2]], [3, [6, 7, 7, 8, 2]]] -> sentence

        
        for word_idx,char_idx in sentence:


            only_words.append(word_idx)
            self.hidden_char=self.dyn_init_hidden(self.char_embedding_dim)
            
            char_idx_tensor = torch.tensor(char_idx).cuda()
            char_embed = self.char_embeddings(char_idx_tensor)
            char_lstm_out,self.hidden_char = self.char_lstm(char_embed.view(len(char_idx_tensor),1,-1), self.hidden_char)
            char_lstm_outputs.append(char_lstm_out[-1])
            
  ## TAKING THE LAST LAYER'S HIDDEN DIM MATRIX
    
        char_lstm_outputs = torch.stack(char_lstm_outputs)

#################################################################################### WORD EMBEDDING PART
        w2v=np.zeros((len(only_words),self.embedding_dim),dtype='float')
        for q,every_wrd in enumerate(only_words):
            v = np.zeros(self.embedding_dim, dtype='float')
            wdr=rev_word_idx[every_wrd]
            try:
                v = word_vectors[wdr]
            except:
                pass
            w2v[q]=v
        
        embeds=torch.cuda.FloatTensor(w2v)
        embeds = embeds.view(len(only_words),1,-1)
###################################################################################### WORD EMBED Ending

#         embeds = self.word_embeds(torch.tensor(only_words).cuda()).view(len(only_words), 1, -1).cuda()
    
        final_input = torch.cat((embeds,char_lstm_outputs),-1)
        lstm_out, self.hidden = self.lstm(final_input, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim*2)

        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).cuda()
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags]).cuda()
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var.cuda() + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


# In[11]:

EMBEDDING_DIM = 200
HIDDEN_DIM = 50
char_embedding_dim = 50



# In[12]:


model = BiLSTM_CRF(len(word_to_ix),len(char_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM,char_embedding_dim)
model=model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)



# In[ ]:

backup_point=-1


# In[ ]:

#model_path='model_chkpoint{}_epochs.pt'.format(backup_point)
#model.load_state_dict(torch.load(model_path))


# In[ ]:

from tqdm import tqdm
from tqdm import tqdm_notebook
# Make sure prepare_sequence from earlier in the LSTM section is loaded

num_epochs=20

for epoch in tqdm(range((backup_point+1),num_epochs),desc='epochs'):  # again, normally you would NOT do 300 epochs, it is toy data
	
    if epoch%5==0:
        try:
            torch.save(model.state_dict(), 'model_chkpoint{}_epochs.pt'.format(epoch))
            print("Checkpoint saved at epoch no.",epoch)
        except Exception as e:
            print("Error in saving the model , ",e)

    if epoch%3==0 and epoch%5!=0:
        try:
            torch.save(model.state_dict(), 'model_chkpoint{}_epochs.pt'.format(epoch))
            print("Checkpoint saved at epoch no.",epoch)
        except Exception as e:
            print("Error in saving the model , ",e)

    for sentence, tags in tqdm(training_data,desc='Trng examples',leave=False):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.

        sentence_in = generate_sequence(sentence, word_to_ix,char_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()




# In[ ]:




# In[ ]:


torch.save(model,'model{}_epochs.pt'.format(num_epochs))


# In[ ]:

torch.save(model.state_dict(),'model_chkpoint{}_epochs.pt'.format(num_epochs))


# In[ ]:

backup_point=23
model_path='model_chkpoint{}_epochs.pt'.format(backup_point)
model.load_state_dict(torch.load(model_path))


# In[ ]:

no_epochs=23
model = torch.load('model{}_epochs.pt'.format(no_epochs))


# In[ ]:

# Check predictions after training

from tqdm import tqdm_notebook

correct=0
total=0
with torch.no_grad():
    for v in tqdm(range(len(test_data))):
        precheck_sent = generate_sequence(test_data[v][0], word_to_ix,char_to_ix)
        precheck_tags = torch.tensor([tag_to_ix[t] for t in test_data[v][1]], dtype=torch.long)
        predicted=model(precheck_sent)
        precheck_tags=precheck_tags.tolist()
        predicted=predicted[1]
        for x,y in zip(predicted,precheck_tags):
            if x==y:
                correct+=1
            total+=1


# In[ ]:

print(correct/total)

 
    




