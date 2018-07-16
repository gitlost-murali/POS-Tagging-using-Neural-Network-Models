
# coding: utf-8

# # Load the torch libraries

# In[1]:

#!/usr/bin/python

import sys, getopt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim


# # Setting device to GPU

# In[2]:

device = torch.device("cuda:0")


# # Encoding the textual data [word1,word2]	 to [word1_idx,word2_idx]

# In[3]:

def get_char_sequence(word,char_to_idx):
	return [char_to_idx.get(i,0) for i in word]

def generate_sequence(seq,word_to_idx,char_to_idx):
#	  print(word_to_idx[seq[0]],get_char_sequence(seq[0],char_to_idx))
	return [[word_to_idx.get(i,0),get_char_sequence(i,char_to_idx)] for i in seq]

def prepare_target(tags, pos_map):
	return autograd.Variable(torch.LongTensor([pos_map.get(i,0) for i in tags]))


# # Loading the data

# In[4]:

import pickle


with open("D:/MyPS3/IIIT H/Chunking/Chunked/total_chunk_tel_data.pickle","rb") as fhnd:
	total_data=pickle.load(fhnd)


# In[5]:

from random import shuffle

shuffle(total_data)


# In[6]:

training_data=total_data[:750]
test_data=total_data[-250:]


# In[7]:

word_to_idx,char_to_idx={},{}	 
pos_map={}

def create_word_char_pos_dicts(totaldata):
	
	tot_data=totaldata[:]	 

	word_to_idx['_unk_']=0
	char_to_idx['_unk_']=0
	pos_map['unk']=0
	for sent,tag in tot_data:
		for j in sent:
			if j not in word_to_idx.keys():
				word_to_idx[j] = len(word_to_idx)

			for k in j:
				if k not in char_to_idx.keys():
					char_to_idx[k] = len(char_to_idx)

		for tg in tag:
			if tg not in pos_map.keys():
				pos_map[tg] = len(pos_map)


# In[ ]:




# In[8]:

create_word_char_pos_dicts(total_data)


# In[9]:

rev_pos_map=dict((v,k) for (k,v) in pos_map.items())


# In[10]:

rev_word_idx=dict((v,k) for (k,v) in word_to_idx.items())


# In[ ]:




# In[7]:

from gensim.models.keyedvectors import KeyedVectors
vectors_file='D:\MyPS3\IIIT H\Pre-Training LM Keras\Large Telugu corpora by ganesh sir\corpus.v3.bin'
word_vectors = KeyedVectors.load_word2vec_format(vectors_file, binary=True, unicode_errors='ignore')


# In[ ]:




# In[ ]:




# In[8]:

word_embedding_dim = 200
char_embedding_dim = 50
hidden_dim = 50


# In[9]:

import numpy as np
class pos_lstm_model(nn.Module):
	def __init__(self,hidden_dim,word_embedding_dim,char_embedding_dim,vocab_size,dist_char_size,pos_tag_size):
		super(pos_lstm_model,self).__init__()
		
		self.hidden_dim = hidden_dim
		self.char_embedding_dim = char_embedding_dim
		
		self.char_embeddings=nn.Embedding(dist_char_size,char_embedding_dim)
		self.char_lstm = nn.LSTM(char_embedding_dim,char_embedding_dim)

		self.word_embeddings=nn.Embedding(vocab_size,word_embedding_dim)

		self.word_lstm = nn.LSTM(word_embedding_dim+char_embedding_dim,hidden_dim)
		
		self.final_tag_layer = nn.Linear(hidden_dim,pos_tag_size)
		
		self.hidden = self.init_hidden(hidden_dim)
		self.hidden_char = self.init_hidden(char_embedding_dim)

	def init_hidden(self,dimension):
		return (torch.zeros(1,1,dimension,device=device),torch.zeros(1,1,dimension,device=device))
	
	def forward(self,sentence):

#  [[0, [0, 1, 2]], [1, [3, 4, 5]], [2, [6, 0, 2]], [0, [0, 1, 2]], [3, [6, 7, 7, 8, 2]]] -> sentence

		only_words = []
		char_lstm_outputs =[]
		
		for word_idx,char_idx in sentence:


			only_words.append(word_idx)
			self.hidden_char=self.init_hidden(self.char_embedding_dim)
			
			char_idx_tensor = torch.tensor(char_idx).cuda()
			char_embed = self.char_embeddings(char_idx_tensor)
			char_lstm_out,self.hidden_char = self.char_lstm(char_embed.view(len(char_idx_tensor),1,-1), self.hidden_char)
			char_lstm_outputs.append(char_lstm_out[-1])
	
			
  ## TAKING THE LAST LAYER'S HIDDEN DIM MATRIX
	
		char_lstm_outputs = torch.stack(char_lstm_outputs)
		
		w2v=np.zeros((len(only_words),word_embedding_dim),dtype='double')
		for q,every_wrd in enumerate(only_words):
			v = np.zeros(word_embedding_dim, dtype='float')
			wdr=rev_word_idx[every_wrd]
			try:
				v = word_vectors[wdr]
			except:
				pass
			w2v[q]=v
		
			
		
#		  word_embed = self.word_embeddings(torch.tensor(w2v).cuda())
		word_embed=torch.cuda.FloatTensor(w2v)
		word_embed=word_embed.float()
		char_lstm_outputs=char_lstm_outputs.float()
		word_embed = word_embed.view(len(sentence),1,-1)
		
		final_input = torch.cat((word_embed,char_lstm_outputs),-1)
		output1,self.hidden = self.word_lstm(final_input,self.hidden)
		tag_pred = self.final_tag_layer(output1.view(len(sentence),-1))
		
		return F.log_softmax(tag_pred,dim=1)


# In[10]:

model = pos_lstm_model(hidden_dim,word_embedding_dim,char_embedding_dim,len(word_to_idx),len(char_to_idx),len(pos_map))
model=model.cuda()

loss_function = nn.NLLLoss().cuda()
OPTIMIZER = optim.SGD(model.parameters(),lr = 0.01)


# In[15]:

no_epochs = 20


# # Training 

# In[16]:

for i in range(no_epochs):	
	
	if i%5 == 0:
		print("epoch ", i, ' of ', no_epochs)
		
	
	for sentence, tags in training_data:

		model.zero_grad()

		model.hidden = model.init_hidden(hidden_dim)

		sentence_in = generate_sequence(sentence, word_to_idx,char_to_idx)
		
		targets = prepare_target(tags, pos_map)
		targets=torch.tensor(targets,dtype=torch.long)
		targets=targets.cuda()
		tag_scores = model(sentence_in)





		loss = loss_function(tag_scores,targets)
		loss.backward()
		OPTIMIZER.step()
print("done")		 


# In[17]:

from pickle import dump


# # Save the model

# In[ ]:

torch.save(model,'model{}_epochs.pt'.format(no_epochs))


# In[ ]:



dump(total_data,open("total_data.pickle","wb"))
dump(training_data,open("training_data.pickle","wb"))
dump(test_data,open("test_data.pickle","wb"))
dump(word_to_idx,open("word_to_idx.pickle","wb"))
dump(char_to_idx,open("char_to_idx.pickle","wb"))
dump(pos_map,open("pos_map.pickle","wb"))
# torch.save(model,'model_pos_tag_char_heirarchial_{}epochs_word_vectors.pt'.format(no_epochs))


# In[11]:

no_epochs=20
model = torch.load('model{}_epochs.pt'.format(no_epochs))


# In[4]:

# model = torch.load('model_pos_tag_char_heirarchial_{}epochs_word_vectors.pt'.format(no_epochs))

from pickle import load
total_data=load(open("total_data.pickle","rb"))
training_data=load(open("training_data.pickle","rb"))
test_data=load(open("test_data.pickle","rb"))
word_to_idx=load(open("word_to_idx.pickle","rb"))
char_to_idx=load(open("char_to_idx.pickle","rb"))
pos_map=load(open("pos_map.pickle","rb"))


# In[5]:

rev_pos_map=dict((v,k) for (k,v) in pos_map.items())


# In[6]:

rev_word_idx=dict((v,k) for (k,v) in word_to_idx.items())


# In[12]:

correct = 0
total = 0
text=''
mismatch_text=''

not_matching=0



with torch.no_grad():
	for k in range(len(test_data)):
		if '' in test_data[k][0]:
			continue
		sentence_in = generate_sequence(test_data[k][0], word_to_idx,char_to_idx)
		
		targets = prepare_target(test_data[k][1], pos_map)
		print(test_data[k][0])
		print(test_data[k][1])
		print(sentence_in)
		tag_scores = model(sentence_in)
#		  print(tag_scores.shape)
#		  print(targets.shape)
		targets=targets.cuda()
		_, predicted = torch.max(tag_scores.data, 1)
#		  print(targets)
#		  print(predicted)
		
		sant_correct=((predicted == targets).sum().item())
		sant_total=targets.size(0)
		total += targets.size(0)
		correct += (predicted == targets).sum().item()
	
		if sant_total!=sant_correct:
			not_matching+=1
#			  print(sentence_in)
			inp_san=[]
			for santance in sentence_in:
				inp_san.append(rev_word_idx[santance[0]])
			print("Sentence is ",inp_san)
			text+="\t".join(inp_san)+'\n'
			inp_tag=[]
			targets=targets.cpu().tolist()
			for nna in targets:
				inp_tag.append(rev_pos_map[nna])
			
			print("Expected is ",inp_tag)
			text+="\t".join(inp_tag)+'\n'
			exp_tag=[]
			predicted=predicted.cpu().tolist()
			
			for nnb in predicted:
				exp_tag.append(rev_pos_map[nnb])
			print("Predicted is ",exp_tag)
			
			text+= "\t".join(exp_tag)+'\n'
			
			comparison=list(np.equal(np.array(predicted),np.array(targets)))
#			  comparison=[y for x in comparison if x==True:y=1 else y=0]
			comparison_li=[]
			y='0'
			for xc in comparison:
				if xc==True:
					y='1'
				else:
					y='0'
				comparison_li.append(y)

			for q,el in enumerate(comparison_li):
				if el=='0':
					try:
						emb_output=word_vectors[inp_san[q]]
						blank_embed="Embedding is present"
					except:
						blank_embed="No embedding"
					mismatch_text+=inp_san[q]+"\t"+inp_tag[q]+"\t"+exp_tag[q]+"\t"+blank_embed+'\n'
			mismatch_text+='\n\n\n'
		

with open('ERROR_ANALYSIS.tsv',"w",encoding='utf-8') as fhn:
	fhn.write(text)		  
	
with open('ERROR_ANALYSIS_WORDS.tsv',"w",encoding='utf-8') as fhn:
	fhn.write(mismatch_text)  

		  
	
	
print('Accuracy of the network on the {} test images: %d %%'.format(len(test_data)) % (100 * correct / total))
print("Error prone sentences are ",not_matching)


# In[13]:

correct/total


# In[1]:




# In[ ]:




# In[ ]:




# # Confusion matrix
# 

# In[14]:

correct = 0
total = 0
text=''
mismatch_text=''
not_matching=0
all_words=[]
all_actual=[]
all_predicted=[]

with torch.no_grad():
	for k in range(len(test_data)):
		if '' in test_data[k][0]:continue
		sentence_in = generate_sequence(test_data[k][0], word_to_idx,char_to_idx)
		
		targets = prepare_target(test_data[k][1], pos_map)
		tag_scores = model(sentence_in)
#		  print(tag_scores.shape)
#		  print(targets.shape)
		targets=targets.cuda()
		_, predicted = torch.max(tag_scores.data, 1)
#		  print(targets)
#		  print(predicted)
		
		sant_correct=((predicted == targets).sum().item())
		sant_total=targets.size(0)
		total += targets.size(0)
		correct += (predicted == targets).sum().item()
		
		
		for santance in sentence_in:
			all_words+=[santance[0]]
			
		targets=targets.cpu().tolist()
		for nna in targets:
			all_actual+=[nna]
			
		predicted=predicted.cpu().tolist()
			
		for nnb in predicted:
			all_predicted+=[nnb]

			
			
		  
		

print('Accuracy of the network on the {} test images: %d %%'.format(len(test_data)) % (100 * correct / total))


# In[15]:

labelsi=[i for i in range(0,len(pos_map))]


# In[16]:

from sklearn.metrics import confusion_matrix, precision_score
np.random.seed(42)


# In[17]:

np.set_printoptions(threshold=np.inf)


# In[ ]:




# In[19]:


cm=np.array(confusion_matrix(all_actual, all_predicted,labels=labelsi))


# In[20]:

print("Confusion Matrix is \n",cm)


# # Include Names to columns and rows

# In[21]:

import pandas as pd


pos_map_rev={val:key for key,val in pos_map.items()}
colnames=[]

for h in range(len(pos_map_rev)):
	colnames.append(pos_map_rev[h])

df=pd.DataFrame(cm,columns=colnames)


# In[22]:

print(df)


# In[23]:

idx = 0
# new_col = [7, 8, 9]  # can be a list, a Series, an array or a scalar	 
df.insert(loc=idx, column='', value=colnames)


# In[24]:

print(df)


# In[25]:

file_name="Confusion Matrix.csv"
df.to_csv(file_name, encoding='utf-8')


# # Calculate Precision and Recall

# In[26]:

# from __future__ import division
precisions={}
recalls={}

for tagname in colnames:
	
	tagindex=pos_map[tagname]
	
	allpredicted=all_predicted.count(tagindex)
	allactual=all_actual.count(tagindex)
	
	true_positives=cm[tagindex][tagindex]
	print(tagname)
	print(true_positives)
	print(allpredicted)
	print(allactual)

	
	if allactual==0:
		recalls[tagname]=0
	elif allactual!=0:
		recalls[tagname]=(true_positives/allactual)
	
	if allpredicted==0:
		precisions[tagname]=0
		
	elif allpredicted!=0:
		precisions[tagname]=(true_positives/allpredicted)
		


# In[27]:

df2=np.zeros((len(pos_map),3),dtype=np.double)
from scipy import stats


# In[28]:

for tagname in colnames:
	
	tagindx=pos_map[tagname]
	precison=precisions[tagname]
	recal=recalls[tagname]
	if precison==0 or recal ==0:
		f_score=0
	else:
		f_score=stats.hmean([ precison , recal ])

	
	df2[tagindx]=[precison,recal,f_score]

	
	


# In[29]:

print(df2)


# In[30]:

table=pd.DataFrame(df2,columns=['precision','recall','f1--score'])


# In[31]:

print(table)


# In[32]:

table.insert(loc=idx, column='', value=colnames)


# In[33]:

table


# In[34]:

file_name="F1score.csv"
table.to_csv(file_name, encoding='utf-8')


# In[ ]:




# In[ ]:



