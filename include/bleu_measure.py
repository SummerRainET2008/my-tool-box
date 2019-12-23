#!/usr/bin/env python

import collections
import functools
import math
import operator
from pa_nlp import common as nlp

try:
  import scipy
  from scipy import array
except ImportError:
  print("Does not find package 'scipy'")

try:
  import tensorflow as tf
except ImportError:
  print("Does not find package 'tensorflow'")

class SentenceBLEU:
  '''
  Regarding corpus BLEU: refer to "nltk.translate.bleu_score.sentence_bleu".
  '''
  def __init__(self, refs: list, max_order=4):
    '''
    :param refs: list[str]
    '''
    
    refs = [ref.split() for ref in refs]
    self._max_order = max_order
    global_ngram2freq = collections.defaultdict()
    for ref in refs:
      ngram2freq = self._stat_ngram_freqs(ref)
      global_ngram2freq = self._cut(global_ngram2freq, ngram2freq)
    self._ref_ngram2freq = global_ngram2freq
    
    self._ref_len = min([len(ref) for ref in refs])

  def compute(self, hyp: str):
    hyp = hyp.split()
    hyp_ngram2freq = self._stat_ngram_freqs(hyp)
    matched = collections.defaultdict(int)
    
    for ngram, freq in hyp_ngram2freq.items():
      order = ngram.count(" ") + 1
      freq = min(freq, self._ref_ngram2freq.get(ngram, 0))
      matched[order] += freq
      
    matched["hyp_len"] = len(hyp)
    
    return self._calc_BLEU(matched)
    
  def _calc_BLEU(self, matched):
    hyp_len, ref_len = matched["hyp_len"], self._ref_len
    precs = [matched[order] / (hyp_len - order + 1 + 1e-10)
             for order in range(1, self._max_order + 1)]
    prec  = functools.reduce(operator.mul, precs, 1) ** (1 / self._max_order)
    ratio = math.exp(min(0, 1 - ref_len / (hyp_len + 1e-10)))
    bleu = prec * ratio
    
    return bleu
  
  def _cut(self, ngram2freq1: dict, ngram2freq2: dict):
    ngram2freq = nlp.defaultdict(int)
    for key in set(list(ngram2freq1.keys()) + list(ngram2freq2.keys())):
      ngram2freq[key] = max(ngram2freq1.get(key, 0), ngram2freq2.get(key, 0))
    return ngram2freq
  
  def _stat_ngram_freqs(self, hyp_list: list):
    ngram2freq = nlp.defaultdict(int)
    for order in range(1, self._max_order + 1):
      for pos in range(0, len(hyp_list) - order + 1):
        ngram2freq[" ".join(hyp_list[pos: pos + order])] += 1
        
    return ngram2freq

def calc_ndcg(rels_list):
  '''relsList: [[1, 2, 3, 1, ...], [2, 3, 1, 0, ..]]
  '''
  def calc_per_dcg(rels):
    rels = rels[: 20] if len(rels) >= 20 else rels + [0] * (20 - len(rels))
    rels = array(rels)
    poses = array(list(range(20)))
    scores = (2 ** rels - 1) / scipy.log(poses + 2)
    return scipy.cumsum(scores)

  def calc_per_ndcg(rels):
    if rels.count(0) == len(rels):
      return array([1.] * 20)

    dcg = calc_per_dcg(rels)
    iDcg = calc_per_dcg(sorted(rels, reverse = True))
    return dcg / iDcg

  ret = sum(map(calc_per_ndcg, rels_list)) / len(rels_list)
  return ret

