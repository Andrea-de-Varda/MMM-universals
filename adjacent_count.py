#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 12:15:48 2021

@author: dev
"""
import plainstream
from transformers import BertTokenizer
import string

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

eng_dump = [] # Indo-European
for line in plainstream.get_text('en', max_words=1000000, tokenize=False):
    line = tokenizer.tokenize(line)
    print(line, "\n\n")
    eng_dump = eng_dump+line 
eng_dump = eng_dump[:1000000]

chi_dump = [] # Sino-Tibetan
for line in plainstream.get_text('zh', max_words=1000000, tokenize=False):
    line = tokenizer.tokenize(line)
    print(line, "\n\n")  
    chi_dump = chi_dump+line
chi_dump = chi_dump[:1000000]

fi_dump = [] # Uralic
for line in plainstream.get_text('fi', max_words=1000000, tokenize=False):
    line = tokenizer.tokenize(line)
    #print(line, "\n\n")
    fi_dump = fi_dump+line
fi_dump = fi_dump[:1000000]

    
def adj_counter_nopunct(text):
    c = 0
    l = len(text)
    for i, t in enumerate(text):
        if i < l-1:
            if t == text[i+1]:
                if t in string.punctuation or t == "[UNK]" or text[i+2][:2] == "##" or t[:2] == "##":
                    pass
                else:
                    #print(text[i-5:i+5], "\t", t)
                    c += 1
    print("N =", c)
            
adj_counter_nopunct(eng_dump) # 19
len(set(eng_dump)) # 24443
(1/24443) # kl
(19/1000000) # 1.9e-05 --> BELOW CHANCE

adj_counter_nopunct(chi_dump) # 1891
len(set(chi_dump)) # 9604
(1/9604) # 0.00010412328196584757
(1891/1000000) # 0.001891 --> above chance

adj_counter_nopunct(fi_dump) # 21
len(set(fi_dump)) # 17546
(1/17546) # 5.699304684828451e-05
(21/1000000) # 2.1e-05 --> BELOW CHANCE


