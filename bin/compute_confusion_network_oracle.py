#!/home3/ly/xiatian/bin/python2.5

'''
    1. Don't know why, in compute_bleu, the formular of bleu should be wrong written so as to get a higher quality oracle!
    2. faint!
    3. only to compute oracle-best bleu for single lattice.
    4. when computing BLEU of partial hypothesis incrementally, don't pose the penalty lead a good result, nearly 
       same to the wrong-written code. 
'''

from sys import argv
from math import log, exp
from operator import mul
from copy import deepcopy       # be carefull on deepcopy
from time import clock
import sys
sys.path.append("/home3/ly/xiatian/include")
from mitel_algorithm import Bleu

class Hyp(object):
    def __init__(self):
        self.boundary = [] 
        self.ngrams = [0, 0, 0, 0, 0]      # ngram matched.
        self.words = {}
        self.length = 0
        self.last_node = None
        self.last_word = None
        self.bleu_score = 0

    def collapse(self, bleu):
        while len(self.boundary) > 0 and " ".join(self.boundary) not in bleu.words:
            self.boundary.remove(self.boundary[0])

    def copy(self):
        hyp = Hyp()
        hyp.boundary = deepcopy(self.boundary)
        hyp.ngrams = deepcopy(self.ngrams)
        hyp.words = deepcopy(self.words)
        hyp.length = self.length
        hyp.last_node = self.last_node
        hyp.lasword = self.last_word
        hyp.bleu_score = self.bleu_score
        return hyp

    def calc_BLEU(self, bleu):
        ngram = [None] * 8 + [bleu.min_ref_len]
        ngram[0], ngram[1] = self.ngrams[1], self.length
        ngram[2], ngram[3] = self.ngrams[2], self.length - 1
        ngram[4], ngram[5] = self.ngrams[3], self.length - 2
        ngram[6], ngram[7] = self.ngrams[4], self.length - 3
        self.bleu_score = Bleu.compute_BLEU(ngram, True)

    def expand(self, word, bleu):
        nhyp = self.copy()
        nhyp.last_node = self
        nhyp.last_word = word 
        if word != "NULL":
            nhyp.boundary.append(word)
            nhyp.collapse(bleu) 
            for ngram in [" ".join(nhyp.boundary[i:]) for i in xrange(len(nhyp.boundary))]:
                nhyp.words[ngram] = nhyp.words.get(ngram, 0) + 1
                if nhyp.words[ngram] <= bleu.words.get(ngram, 0):
                    nhyp.ngrams[len(ngram.split())] += 1
            if len(nhyp.boundary) == 4:
                nhyp.boundary.remove(nhyp.boundary[0])
            nhyp.length += 1
            nhyp.calc_BLEU(bleu)
        return nhyp

class Decoder(object):
    def __init__(self, lattice, bleu):
        self.lattice = lattice
        self.bleu = bleu

    def decode(self):
        cky = [[Hyp()]]
        for edge_id, edge in enumerate(self.lattice):
            print len(cky[-1]), "hyps would extend edge set:", edge
            mp = {}
            for hyp in cky[-1]:
                for w in edge:
                    nhyp = hyp.expand(w, self.bleu)
                    bdy = "_".join(nhyp.boundary)
                    if bdy not in mp:
                        mp[bdy] = nhyp
                    elif nhyp.bleu_score > mp[bdy].bleu_score:
                        mp[bdy] = nhyp
            nowhyps = mp.values()
            nowhyps.sort(key = lambda x: -x.bleu_score)
            cky.append(nowhyps[0: min(500, len(nowhyps))])
        opt_hyp = max(cky[-1], key = lambda x: x.bleu_score)
        print "opt_hyp", opt_hyp, opt_hyp.bleu_score
        print
        solution = []
        while True: 
            solution.append(opt_hyp.last_word)
            opt_hyp = opt_hyp.last_node
            if solution[-1] is None:
                break
        solution = list(reversed(solution[: -1]))
        return solution

def get_lattice(fin):
    while True:
        try:
            ln = fin.next()
        except StopIteration:
            break
        if not ln.startswith("ave_TER"):
            continue
        lattice = []
        span_num = int(fin.next().split()[1])
        for span_id in xrange(span_num):
            key_num = int(fin.next().split()[1])
            keys = [fin.next().split()[0] for key_id in xrange(key_num)]
            keys.sort()
            lattice.append(keys)
        yield lattice

if __name__ == "__main__":
    try:
        import psyco
        psyco.full()
        print "optimization on"
    except:
        print "optimization off"

    assert len(argv) == 5, "cmd lattice ref out mask_sys"
    argv[3] += ".oracle"
    mask_sys = int(argv[4])

    fin = file(argv[1], "rU")
    src_num = int(fin.next().split()[1])
    sys_num = int(fin.next().split()[1])
    refs = [ln[ln.find(">") + 1: ln.rfind("<")] for ln in file(argv[2], "rU") if ln.startswith("<seg")]
    assert len(refs) == src_num * 4

    fous = {}
    mask_syses = range(sys_num) if mask_sys == -1 else [mask_sys]
    for sys_id in mask_syses:
        fous[sys_id] = file("%s.%d" %(argv[3], sys_id), "w")
                
    fou = file(argv[3], "w")
    print >> fou, src_num
    print >> fou, sys_num
    for sys_id in mask_syses:
        print >> fous[sys_id], src_num

    for src_id in xrange(src_num):
        bleu = Bleu([refs[src_id + i * src_num] for i in xrange(4)])
        print >> fou, src_id
        for sys_id in xrange(sys_num):
            lattice = get_lattice(fin).next()
            if sys_id not in mask_syses:
                continue
            print "src, sys:", src_id, sys_id
            start = clock()
            decoder = Decoder(lattice, bleu)
            solution = decoder.decode()
            finish = clock()

            print "time:", finish - start
            assert len(lattice) == len(solution)
            print >> fous[sys_id], "src:"
            print >> fous[sys_id], "1"
            print >> fous[sys_id], " ".join([w for w in solution if w != "NULL"])
            ssolutuon = ["%d:%d-%s" %(len(nd), nd.index(s), s) for [nd, s] in zip(lattice, solution)]
            print >> fou, " ".join(ssolutuon)

    for sys_id in mask_syses:
        fous[sys_id].close()
    fou.close()        

    print "all is OK"
