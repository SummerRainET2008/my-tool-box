#!/home3/ly/xiatian/bin/python2.5

'''
    To transform the prediction from CRF to a formal nbest, so as to be computed the BLEU.
'''

from sys import argv
from math import log, exp
from operator import mul
from copy import deepcopy
from compute_combined_oracle import get_nbest

if __name__ == "__main__":
    '''try:
        import psyco
        psyco.full()
        print "optimization on"
    except:
        print "optimization off"'''

    assert len(argv) == 3, "cmd combined_oracle nbest.out"

    nbests = list(get_nbest(argv[1]))
    src_num = len(nbests)
    print "src_num:", src_num
    fou = file(argv[2], "w")

    for src_id in xrange(src_num):
        solu = int(nbests[src_id][0].split("|||")[-1])
        feats = " ".join(["%d:%s" %(ind, w) for ind, w in enumerate(nbests[src_id][1 + solu].split("|||")[1].split())])
        print >> fou, solu, feats
    fou.close()   
    print "all is OK"

