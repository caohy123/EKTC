from collections import Counter
import numpy as np

def evaluate_dist_n(data, n):
    return _distinct_n(data, n)

def get_ngrams(resp, n):
    tokens = resp.split()
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-(n-1))]

def get_ngram_counter(resp, n):
    ngrams = get_ngrams(resp, n)
    counter = Counter()
    counter.update(ngrams)
    return counter

def _distinct_n(data, n):
    dist_results = []
    for sent in data:
        ngram_counter = get_ngram_counter(sent.strip().lower(), n)
        
        if sum(ngram_counter.values()) == 0:
            print("Warning: encountered a response with no {}-grams".format(n))
            print(sent.strip().lower())
            print("ngram_counter: ", ngram_counter)
            continue
        
        dist = len(ngram_counter) / sum(ngram_counter.values())
        dist_results.append(dist)
    
    return np.average(dist_results)