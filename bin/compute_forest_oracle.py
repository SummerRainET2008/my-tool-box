#!/usr/bin/python

__date__ = "2009/12/12"

USER_INCLUDE_PATH = ["/bwdata157/xiatian/Include"]

import sys
from sys import argv
from math import log, exp, fabs
from operator import mul
from copy import deepcopy       # be carefull on deepcopy
from time import clock

if USER_INCLUDE_PATH not in sys.path:
        sys.path.extend(USER_INCLUDE_PATH)
from mitel_algorithm import Bleu

sys_num     = None 
mask_sys_id = None
src_num     = None
debug_level = 0

class Hyp(object):
    def __init__(self, token):
        f, t = token.find(":"), token.rfind(":")
        tokens = [token[: f], token[f + 1: t], token[t + 1:]]
        self.last_hyp_id = int(tokens[0])
        self.last_word   = tokens[1]
        self.neighbour   = None
        self.boundary    = []
        self.ngrams      = [0] + [0] * 4
        self.words       = {}
        self.bleu_score  = 0
        self.length      = 0
 
class Decoder(object):
    def compute_oracle(self, forest_fe, ref_fe, out_fe):
        global src_num
        global sys_num
        forest_iter = self.read_forest(forest_fe)
        src_num = forest_iter.next()
        sys_num = forest_iter.next()
        bleu_iter = self.analyze_bleu(ref_fe)

        fou = file(out_fe, "w")
        print >> fou, src_num
        for src_id in xrange(src_num):
            print "src:", src_id
            print >> fou, 1
            bleu = bleu_iter.next()
            forests = [self.get_oracle_best(forest, bleu) + [id] for id, forest in enumerate(forest_iter.next())]
            max_forest = max(forests, key = lambda x: x[1])
            if mask_sys_id != -1:
                max_forest[2] = mask_sys_id
            print >> fou, "%s ||| %f ||| %d" %(max_forest[0], max_forest[1], max_forest[2])
        fou.close()           

    def analyze_bleu(self, ref_fe):
        refs = [ln[ln.find(">") + 1: ln.rfind("<")] for ln in file(ref_fe, "rU") if ln.startswith("<seg")]
        assert len(refs) % src_num == 0, "%d % %d" %(len(refs), src_num)
        rev_num = len(refs) / src_num
        for src_id in xrange(src_num):
            rs = [refs[t * src_num + src_id] for t in xrange(rev_num)]
            yield Bleu(rs)

    def get_oracle_best(self, forest, bleu):
        for span_id in xrange(1, len(forest)):
            for hyp_id, hyp in enumerate(forest[span_id]):
                hyps = []
                while hyp is not None:
                    self.compute_score(hyp, forest[span_id - 1][hyp.last_hyp_id], bleu)
                    hyps.append(hyp)
                    hyp = hyp.neighbour
                hyps.sort(key = lambda x: -x.bleu_score) 
                for ind, hyp in enumerate(hyps[: -1]):
                    hyp.neighbour = hyps[ind + 1]
                hyps[-1].neighbour = None
                forest[span_id][hyp_id] = hyps[0]
        best_hyp = max(forest[-1], key = lambda x: x.bleu_score)
        bleu = best_hyp.bleu_score
        tran = []
        for span_id in xrange(len(forest) - 1, 0, -1):
            if best_hyp.last_word != "NULL":
                tran.append(best_hyp.last_word)
            best_hyp = forest[span_id - 1][best_hyp.last_hyp_id]              
        tran.reverse()            
        return [" ".join(tran), bleu]

    def compute_score(self, hyp, last_hyp, bleu):
        hyp.words = deepcopy(last_hyp.words)
        hyp.boundary = deepcopy(last_hyp.boundary)
        hyp.ngrams = deepcopy(last_hyp.ngrams)
        if hyp.last_word == "NULL":
            hyp.length = last_hyp.length
            hyp.bleu_score = last_hyp.bleu_score
        else:
            hyp.length = last_hyp.length + 1
            hyp.boundary.append(hyp.last_word)
            for i in xrange(len(hyp.boundary)):
                w = " ".join(hyp.boundary[i:])
                if w in bleu.words and hyp.words.get(w, 0) + 1 <= bleu.words[w]:
                    hyp.words[w] = hyp.words.get(w, 0) + 1
                    hyp.ngrams[len(hyp.boundary) - i] += 1
            if len(hyp.boundary) == 4:
                hyp.boundary.remove(hyp.boundary[0])
            hyp.bleu_score = Bleu.compute_BLEU(hyp.ngrams, hyp.length, bleu.min_ref_len, True)

    def read_forest(self, fe):
        tokiter = iter(file(fe, "rU").read().split())
        tokiter.next()
        yield int(tokiter.next())
        tokiter.next()
        yield int(tokiter.next())

        for src_id in xrange(src_num):
            forests = []
            for sys_id in xrange(sys_num):
                while tokiter.next() != "span_size:":
                    pass
                span_size = int(tokiter.next())
                forest = []
                for span_id in xrange(span_size):
                    span_forst = []
                    _, _ = tokiter.next(), tokiter.next()       #span_id: 0
                    _, beam_size = tokiter.next(), int(tokiter.next())
                    while beam_size > 0:
                        beam_size -= 1
                        span_forst.append(Hyp(tokiter.next()))
                        tail = span_forst[-1]
                        while True:
                            tag = tokiter.next()
                            if tag == "end":
                                break
                            tail.neighbour = Hyp(tag)
                            tail = tail.neighbour
                    forest.append(span_forst)
                if mask_sys_id == -1 or sys_id == mask_sys_id:
                    forests.append([[Hyp("0:headnode:0")]] + forest)
            yield forests                    

if __name__ == "__main__":
    assert len(argv) == 4, "cmd forest ref mask_sys[-1,0..sys_num - 1]"

    mask_sys_id = int(argv[3])
    d = Decoder()
    d.compute_oracle(argv[1], argv[2], "%s.oracle.sys%d" %(argv[1], mask_sys_id))

    print "all is OK"


