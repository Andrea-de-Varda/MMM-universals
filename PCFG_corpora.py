#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:03:22 2021

@author: dev
"""
from pcfg import PCFG
import numpy as np
from itertools import permutations, chain
import statistics
import random
from collections import Counter

vocabulary = []
for p in permutations([l for l in "abcdefghijklmnopqrstuvwxyzàèùò123456789"], 3):
    vocabulary.append("".join(p))
print(len(vocabulary))
vocabulary = vocabulary[:50000] # starting with  50.000 

# to assign probabilities of terminal symbols according to the Zipf's Law
def zipf_p(n):
    s = np.random.zipf(2, n) # where 2 is a, distribution parameter
    s_p = s/sum(s)
    rest = 1-sum(s_p) # for the PCFG, the probabilities need to sum to 1, exactly --> correcting for rounding errors
    for i, j in enumerate(s_p):
        if j == max(s_p):
            s_p[i] = j + rest
            break
    if sum(s_p) != 1:
        raise ValueError("The distribution does not sum to 1")
    return s_p

###################
# nested brackets #
###################

p1 = zipf_p(len(vocabulary))
p = p1/2.5 # 1/2.5 = 0.4 --> actually, we want the probabilities to sum to 0.4, with 0.6 p of empty string
p_empty = 1-sum(p) # 0.6 +- rounding errors

template_S = '"{0}" S "{0}" S [{1}]' # --> "abc" S "abc" S [0.2]
temp = []
for v, prob in zip(vocabulary, list(p)):
    temp.append(template_S.format(v, str(prob)))
S = " | ".join(temp)
    
template_grammar = "S -> {0} | [{1}]"
grammar = PCFG.fromstring(template_grammar.format(S, str(p_empty)))

nested_sentences = []
for sentence in grammar.generate(1000000): # for now, 1,000,000, but then remove empty strings
    print(sentence)
    nested_sentences.append(sentence)
nested_sentences = list(filter(None, nested_sentences)); print(len(nested_sentences)) # removed empty strings
nested_sentences = nested_sentences[:1000] # subset of 1,000 sentences

mean_length = statistics.mean([len(sentence.split()) for sentence in nested_sentences]); print(mean_length)
std_length = statistics.stdev([len(sentence.split()) for sentence in nested_sentences]); print(std_length)

###################
# "flat" brackets # 
###################
# To ensure that average distance between opening and closing bracket matches (Papadimitriou & Jurafsky did not do that), simply shuffle nested brackets

flat_sentences = []
for sentence in nested_sentences:
    s = sentence.split()
    random.shuffle(s)
    s = " ".join([item for item in s])
    flat_sentences.append(s)
    
#################
# random corpus #
#################
# We want all the corpora to be composed by the same number of sentences, each composed by the same number of tokens.

random_sentences = []
for sentence in nested_sentences:
    l = len(sentence.split())
    temp = []
    for n in range(l):
        samp = random.sample(vocabulary, 1)[0] 
        temp.append(samp)
    random_sentences.append((" ".join(temp)))

###############
# random Zipf #
###############

zipf_sentences = []
for sentence in nested_sentences:
    l = len(sentence.split())
    s = list(np.random.choice(vocabulary, l, p=p1)) # choose l items according to previously defined p1 
    zipf_sentences.append((" ".join(s))) 
    
# check if focabulary is Zipf-distributed
words = []
for sentence in zipf_sentences:
    words.append(sentence.split())
words = list(chain.from_iterable(words))  
freq = dict(Counter(words))
dict(sorted(freq.items(), key=lambda item: item[1]))

# saving all files to .txt
for name, list_ in zip(["nested_sentences", "flat_sentences", "random_sentences", "zipf_sentences"], [nested_sentences, flat_sentences, random_sentences, zipf_sentences]):
    with open(name+'.txt', 'w') as f:
        for item in list_:
            f.write("%s\n" % item)
