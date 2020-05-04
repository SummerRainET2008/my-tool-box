#!/usr/bin/env python

from algorithm import *  

def BLEU(ngram, sent_bleu = True):
  '''ngram: should be float[10].
      1. If to compute SBP_BLEU, hyp_length is the min(hyp_length, ref_length) 
      or hyp_length is the true length of hypothesis.
      2. when computing BLEU on sentence level, it's better to throw penalty 
      factor in practice.
  '''
  assert len(ngram) == 10
  precs = [ngram[i * 2] / float(ngram[i * 2 + 1]) + 1e-10 for i in xrange(4)]
  prec  = reduce(mul, precs, 1) ** 0.25
  ratio = 1 if sent_bleu \
      else exp(min(0, 1 - ngram[8] / float(ngram[9] + 1e-10)))
  bleu = prec * ratio
  return bleu

class Hyp(object):
  SBP_BLEU    = False     # need to set new value.
  FEAT_NUM    = None
  INI_WEIGHT    = None

  __slots__ = ["tran", "ngram", "feats", "bleu", "score"]

  def __init__(self, ln):
    toks = ln.split("|||")
    self.tran  = toks[0]
    ngram = map(float, toks[3].split()) + [None]
    ngram[9] = min(ngram[1], ngram[8]) if Hyp.SBP_BLEU else ngram[1]
    self.ngram = array(ngram)                  # float[10]
    self.feats = self._analyze_feats(toks[1])
    self.bleu = None
    self.score = self.update_score(Hyp.INI_WEIGHT) 
     
  def _analyze_feats(self, featstr): # only appliable in MY FORMAT.
    feats = {}
    for tok in featstr.split():
      fid, v = tok.split(":")
      feats[int(fid)] = float(v)
    return feats

  def update_score(self, weight):
    self.score = sum([weight[fid] * v for fid, v in self.feats.items()])
    return self.score

class Data(object):
  MAX_SRC     = 1024 * 1024 

  @staticmethod
  def get_nbest(fn):
    fin = open(fn)
    src_num = min(int(fin.next()), Data.MAX_SRC)
    yield src_num
    for sid in xrange(src_num):
      fin.next()
      hyp_num, pdata = int(fin.next()), []
      for hid in xrange(hyp_num): 
        hyp = Hyp(fin.next())
        pdata.append(hyp)
      yield pdata    

  @staticmethod
  def read_data(fn_nbests):
    start = time()
    data = []
    fiters = [Data.get_nbest(fn) for fn in fn_nbests.replace(";", " ").replace(",", " ").split()]
    mynext = lambda: map(methodcaller("next"), fiters)
    src_num = int(mynext()[0])
    for sid in xrange(src_num):
      hyps = sum(mynext(), [])
      hyps.sort(key = lambda h: -h.score)
      dup = set()
      pdata = []
      for hyp in hyps:
        if (hyp.tran, int(hyp.score * 1e6)) not in dup:
          hyp.bleu = BLEU(hyp.ngram, True)  # +1 smoothing
          pdata.append(hyp)
          dup.add((hyp.tran, int(hyp.score * 1e6)))
      print "Sid: %d, total number: %d, the number of duplicate hyptheses: %d" %(sid, len(pdata), len(hyps) - len(pdata))
      data.append(pdata)
  
    Data._compute_oracle(data)
    print "memory: %f MB, time: %f seconds" %(get_memory(), time() - start)
    return data     

  @staticmethod
  def _compute_oracle(data):
    top1_ngram, oracle_ngram = array([0.0] * 10), array([0.0] * 10)
    for ind, pdata in enumerate(data): 
      top1_ngram += pdata[0].ngram
      bh = max(pdata, key = lambda h: h.bleu)
      oracle_ngram += bh.ngram
    
    print "SBP_BLEU" if Hyp.SBP_BLEU else "BLEU"  
    top1_bleu, oracle_bleu = BLEU(top1_ngram, False), BLEU(oracle_ngram, False)
    print "top1: %f oracle: + %f" %(top1_bleu, oracle_bleu - top1_bleu)

if __name__ == "__main__":
  parser = OptionParser(usage = "cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose", default = False, help = "don't print status messages to stdout")
  parser.add_option("--nbests",     dest = "fn_nbests",   default = None,   help = "multi-nbest files are supported")
  parser.add_option("-w",       dest = "fn_weight",   default = None,   help = "")
  parser.add_option("--feat_num",   dest = "feat_num",    default = 7491,   type = "int", help = "default 7491")
  (options, args) = parser.parse_args()
  start = time()

  Hyp.FEAT_NUM = options.feat_num
  Hyp.UFEAT = array([0.0] * Hyp.FEAT_NUM)
  #Data.MAX_SRC = 124
  Hyp.SBP_BLEU = False 
  if options.fn_weight is not None:
    Hyp.INI_WEIGHT = array(map(float, open(options.fn_weight).read().split()))

  data = Data.read_data(options.fn_nbests)
