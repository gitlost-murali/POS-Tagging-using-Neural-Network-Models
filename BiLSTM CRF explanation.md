
BilSTM crf explanation


sentence is  tensor([ 11,  12,  13,  14,  15,  16,  11])  # Encoded sentence (Tokenized)

### Input features to CRF is the output of BiLSTM (no.of words,no.of tags).Here no of tags is 5.

Features is
tensor([[ 0.5097, -0.0811,  0.3995, -0.1383,  0.3875],
        [ 0.4825, -0.0873,  0.4140, -0.1281,  0.4278],
        [ 0.4511,  0.0929,  0.1660, -0.2906,  0.4013],
        [ 0.5103, -0.0333,  0.2920, -0.2044,  0.3373],
        [ 0.4635, -0.0438,  0.2888, -0.2380,  0.3425],
        [ 0.2789, -0.0071,  0.2396, -0.3721,  0.4266],
        [ 0.3305,  0.1088,  0.0476, -0.3885,  0.4252]])

#### These features can also be called emission probabilities

Initial alpha = tensor([[-10000., -10000., -10000., -10000., -10000.]])
Forward_var = tensor([[-10000., -10000., -10000.,      0., -10000.]])

## Useful for backtracking. Just keep this in mind. You will understand its use soon.

Forward var beginning is  tensor([[-10000., -10000., -10000.,      0., -10000.]])

forward tensors aplhas_t  []


## Here feat means feature(output of a word from BiLSTM (1,#tags))


## For every feat, all the tags can occur with some probabilities. Let's calculate them

probabilities are found by adding "emission score","transmission score" .
"transmission score" is a matrix of (#tags,#tags) with elements showing transition scores from each other.
Notation is a little different .

**** Entry i,j is the score of # transitioning *to* i *from* j.  **

feat is  1
  For tag  0
  Before addition scores are :

  1.forward_var:  tensor([[-10000., -10000., -10000.,      0., -10000.]])
  2.trans_score:  tensor([[   -1.7072,    -0.8968,    -0.2316,    -0.6008, -9999.9902]])   # transmission_score[0] gives score of # transitioning *to* i from all states.
  3.Emit score : tensor([[ 0.5097,  0.5097,  0.5097,  0.5097,  0.5097]])                   # We take emission scores[ith word,0th tag]

  # We keep same scores as we are finding all these probabilities for one word.

  Summation is  tensor([[-1.0001e+04, -1.0000e+04, -9.9997e+03, -9.1082e-02, -1.9999e+04]])

  forward tensors aplhas_t  [tensor(1.00000e-02 *[-9.1082])]                ## This is simply doing log_sum(Summation)

  ## So , now we have score of tag1 for word1 .

  For tag  1
  Before addition scores are :

  1.forward_var:  tensor([[-10000., -10000., -10000.,      0., -10000.]])
  2.trans_score:  tensor([[   -0.6801,     1.2532,     0.6238,     1.5398, -9999.9902]])
  3.Emit score : tensor(1.00000e-02 *
         [[-8.1111, -8.1111, -8.1111, -8.1111, -8.1111]])
  Summation is  tensor([[-10000.7607,  -9998.8281,  -9999.4570,      1.4587, -20000.0723]])
  forward tensors aplhas_t  [tensor(1.00000e-02 *[-9.1082]), tensor([ 1.4587])]

  ## So , now we have score of tag2 for word1 . You can see that "forward_var" score is same for the above one too. The only thing that changes is
   emission score since Emit score=emission scores[ith word,1st tag].


  For tag  2
  forward tensors aplhas_t  [tensor(1.00000e-02 *[-9.1082]), tensor([ 1.4587]), tensor([-1.1913])]

  For tag  3
  forward tensors aplhas_t  [tensor(1.00000e-02 *[-9.1082]), tensor([ 1.4587]), tensor([-1.1913]), tensor([-10000.1289])]

  For tag  4
  forward tensors aplhas_t  [tensor(1.00000e-02 *[-9.1082]), tensor([ 1.4587]), tensor([-1.1913]), tensor([-10000.1289]), tensor([-0.7333])]

Now , we update our "Forward var". We use this "Forward_var" for word2.

  Forward var= aplhas_t
  Forward var now is  tensor([[-9.1082e-02,  1.4587e+00, -1.1913e+00, -1.0000e+04, -7.3327e-01]])

  forward tensors aplhas_t  []



feat is  2
  For tag  0
  Before addition scores are :

  1.forward_var:  tensor([[-9.1082e-02,  1.4587e+00, -1.1913e+00, -1.0000e+04, -7.3327e-01]])
  2.trans_score:  tensor([[   -1.7072,    -0.8968,    -0.2316,    -0.6008, -9999.9902]])
  3.Emit score : tensor([[ 0.4825,  0.4825,  0.4825,  0.4825,  0.4825]])
  Summation is  tensor([[-1.3158e+00,  1.0443e+00, -9.4035e-01, -1.0000e+04, -1.0000e+04]])
  forward tensors aplhas_t  [tensor([ 1.2528])]


.. And so on


## Finally, after processing the last word, we get "Forward_var".

Forward var now is  tensor([[   10.6157,    10.3775,    11.8779, -9988.8350,    10.5265]])

Alpha (Our score is)  tensor(11.9781)   ## We do log_sum_exp


Golden score is calculated simply by adding emission and transition scores for the known path.

Loss function is Our score-Golden score

Backprop , you're done.

Will soon update regarding viterbi_decode (Inference).

For any doubts, contact me on kmanoharmurali@gmail.com, +91-9959440709
