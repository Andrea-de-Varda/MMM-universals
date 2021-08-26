# MMM-universals
Materials and code relative to the MSc thesis "Language Universals in Massively Multilingual Models: an Empirical Investigation" (2021).

### Iterative token-level cloze task (ITCT) - Chapter 3
The ITCT consists in an adaptation of mBERT's native functionality, i.e. **masked language modeling** (MLM). The main difference consists in the fact that while in MLM the model has to predict the tokens corresponding to the masks applied to a randomly selected subpart of the input (15% of the tokens in the sentence), in the ITCT all the tokens are masked iteratively. This mitigates the aleatory dimension in the selection of the tokens that are masked, and provides a comprehensive metric of the **probability assigned to a sentence**, where each token has to be predicted by the model given the whole remaining context.  After freezing its weights, in the first timestep t0 the model is presented with the input sentence where the first token is masked, i.e. substituted with a special mask token, and two special characters -- `[CLS]` and `[SEP]` -- are appended to the beginning and the end of the sequence, to mark the sentence boundaries. The model then predicts the original token relying the right context; the hidden vector corresponding to the masked token is passed through a softmax over the vocabulary, in order to assign a probability to the masked token. At t1, the mask is moved from the first to the second token, and now mBERT's prediction is conditioned by both the right and the left context. The process is repeated until the end of the sentence. 

    1. [MASK] should buy a car
    2. John [MASK] buy a car
    3. John should [MASK] a car
    4. John should buy [MASK] car
    5. John should buy a [MASK]

The file `ITCT.py` receives as input a text file where each sentence is separated by a new line. It returns the overall probability assigned to each sentence by BERT, mBERT, and BERT initialized with random weights in different output text files (mono_mean_p.txt, multi_mean_p.txt, random_mean_p.txt) located in a folder named as the txt file in input. 

`python3 ITCT.py -c input_file.txt`

### Corpora generated through PCFG - Section 4.1
The file `PCFG_corpora.py` contains the code to build the four artificial corpora employed in our first experiment (nested brackets, flat brackets, zipf corpus, random corpus).

### Adjacent tokens count - Section 4.4.1
The file `adjacent_count.py` counts the amount of identical adjacent tokens in Wikipedia dumps in English, Chinese, and Finnish.


