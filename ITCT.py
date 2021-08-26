#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 12:16:27 2021

@author: dev
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:07:34 2021

@author: dev
"""
# ITCT code
# run it as: python3 terminal_loss_p.py -c try.txt
import argparse
import os
import re
import statistics
import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig # BertModel

parser = argparse.ArgumentParser(description='Derive metrics for corpora')
parser.add_argument('-c', action='store', dest='simple_value', help='Store corpus destination')
results = parser.parse_args()


tokenizer_multi = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model_multi = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
model_multi.eval()

tokenizer_mono = BertTokenizer.from_pretrained('bert-base-cased')
model_mono = BertForMaskedLM.from_pretrained('bert-base-cased')
model_mono.eval()

config = BertConfig.from_pretrained('bert-base-multilingual-cased')
model_random = BertForMaskedLM(config)
model_random.eval()

sm = torch.nn.Softmax(dim=0)
torch.set_grad_enabled(False)

dirName = re.sub("\.txt", "", results.simple_value)
try:
    os.mkdir(dirName)
    print("Directory " , dirName ,  "created") 
except FileExistsError:
    print("Directory " , dirName ,  "already exists")

def list_to_txt(l, filename):
    filename = re.sub("\.txt", "", filename)
    with open(dirName+"/"+filename+".txt", 'w') as f:
        for item in l:
            f.write("%s\n" % item)

def get_p(sentence, lang):
    if lang == "mono":
        tokenizer = tokenizer_mono
        model = model_mono
    if lang == "multi":
        tokenizer = tokenizer_multi
        model = model_multi
    if lang == "random":
        tokenizer = tokenizer_multi
        model = model_random
    tokens = tokenizer.tokenize(sentence)#, return_tensors='pt')
    p = []
    for n in range(len(tokens)):
        sent = tokens.copy()
        mask_filler = sent[n]
        sent[n] = "[MASK]"
        tokenize_input = ["[CLS]"]+sent+["[SEP]"]
        #print(tokenize_input, mask_filler)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        masked_position = (tensor_input.squeeze() == tokenizer.mask_token_id).nonzero().item()
        output = model(tensor_input)
        last_hidden_state = output[0].squeeze(0)
        # only get output for masked token (output is the size of the vocabulary)
        mask_hidden_state = last_hidden_state[masked_position]
        # convert to probabilities (softmax) --> giving a probability for each item in the vocabulary
        probs = sm(mask_hidden_state)
        target_id = tokenizer.convert_tokens_to_ids(mask_filler)
        #print(probs[target_id].item())
        p.append(probs[target_id].item())
    mean_p = statistics.mean(p)
    #print("Mean P =", mean_p)
    return p, mean_p

# Now using Native Loss!

def p_corpus(name, lang):
    corpus = open(name).readlines()
    P = []
    P_all = []
    for line in corpus:
        p = get_p(line, lang)
        P.append(p[1])
        P_all.append(p[0])
    P_all = [i1 for i in P_all for i1 in i]
    print(name, lang, "P =", statistics.mean(P))
    return P, P_all

for language in ["multi", "mono", "random"]:
    output = p_corpus(results.simple_value, language)
    list_to_txt(output[0], language+"_mean_p")
    list_to_txt(output[1], language+"_all_p")