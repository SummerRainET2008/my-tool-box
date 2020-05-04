#!/home3/ly/xiatian/bin/python2.5

'''
    1. To compute the oracle selecting the best hypothesis from oracler systems.
    2. By default, we choose the first hypothesis from each system to join together.
'''

import sys
from sys import argv
from math import log, exp
from operator import mul
from copy import deepcopy

USER_INCLUDE_PATH = ["/home3/ly/xiatian/include"]
if USER_INCLUDE_PATH not in sys.path:
    sys.path.extend(USER_INCLUDE_PATH)

from mitel_algorithm import Bleu

src_num         = None
sys_num         = 4
per_nbest_num   = sys.maxint # for CRF experiments, let this value be 1, or sys.maxint

def get_TER(fe):
    global src_num
    ters = [log(float(line.split()[1])) for line in file(fe, "rU") if line.startswith("ave_TER")]
    assert len(ters) % sys_num == 0, "ters: %d" %len(ters)
    src_num = len(ters) / sys_num
    for src_id in xrange(src_num):
        ter = ters[: 4]
        ters = ters[4:]
        sm = sum(ter)
        ter = [t / sm for t in ter]
        yield ter

def get_nbest(fe):
    fin = file(fe, "rU")
    src_num = int(fin.next())
    for src_id in xrange(src_num):
        fin.next()
        hyp_num = int(fin.next())
        hyps = []
        for i in xrange(hyp_num):
            hyp = fin.next().strip().split("|||")
            hyps.append("|||".join(hyp))
        yield hyps

if __name__ == "__main__":
    assert len(argv) >= 4, "cmd ref nbest-num [nbest0 nbest1 ..] oracle.out"

    refs = [line[line.find(">") + 1: line.rfind("<")] for line in file(argv[1], "rU") if line.startswith("<seg")]
    assert len(refs) % 4 == 0 and len(refs) > 0
    src_num = len(refs) / 4

    # generate oracle.
    nbest_num = int(argv[2])
    nbest_fes = argv[3: 3 + nbest_num]
    nbestiter = [get_nbest(fe) for fe in nbest_fes] 
    print len(nbestiter)
    foracle = file(argv[nbest_num + 3], "w")                        
    print >> foracle, src_num
    sta_data = [0] * nbest_num
    for src_id in xrange(src_num):
        print "src_id", src_id
        print >> foracle, "src_ID:", src_id
        print >> foracle, nbest_num + 1
        bleu = Bleu([refs[src_num * i + src_id] for i in xrange(4)])
        hyps = []
        best = [None, None, None]   # bleu, hyp, sys_id
        onebests = []
        for sys_id, nit in enumerate(nbestiter):
            nbests = nit.next()
            nbests = nbests[0: min(per_nbest_num, len(nbests))]   
            onebests.append(nbests[0])
            hyps = [(Bleu.compute_hyp_BLEU(h.split("|||")[0], bleu), h) for h in nbests]
            besthyp = max(hyps)
            if best[0] is None or besthyp[0] > best[0]:
                best = besthyp[0], besthyp[1], sys_id
        print >> foracle, "%s ||| %s ||| %d" %(best[1], best[0], best[2])
        sta_data[best[2]] += 1
        for ind, hyp in enumerate(onebests):
            hyp = hyp.split("|||")
            print >> foracle, "|||".join(hyp), "|||", Bleu.compute_hyp_BLEU(hyp[0], bleu)
    foracle.close()
    
    sta_data = [(float(num) / src_num, sys_id) for sys_id, num in enumerate(sta_data)]
    sta_data.sort(key = lambda x: -x[0])
    for rate, sys_id in sta_data:
        print "file = %s, \trate = %f" %(nbest_fes[sys_id], rate)
    print "all is OK"

        


