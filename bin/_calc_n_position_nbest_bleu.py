#!/usr/bin/env python

'''Give the statistics of the N-th position in a N-best list.
   1. The N-best list should be ordered.
'''

import sys
if "/nfs/18/wsu0215/include" not in sys.path:
    sys.path.append("/nfs/18/wsu0215/include")
from algorithm import *
from bleu_score_v1 import *

if __name__ == "__main__":
    parser = OptionParser(usage = "cmd [optons] ..]")
    #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose", default = False, help = "don't print status messages to stdout")
    parser.add_option("-c", "--cand", dest = "fn_cand", default = "cand", help = "")
    parser.add_option("-f", "--feat", dest = "fn_feat", default = "feat", help = "")
    parser.add_option("-w",           dest = "fn_ws",   default = None,   help = "initial weights, None by default")
    (options, args) = parser.parse_args()

    #Data.MAX_SRC = 10
    Hyp.SBP_BLEU = False 
    Hyp.ORACLE_TYPE = 3
    data = Data.read_data(options.fn_cand, options.fn_feat, options.fn_ws)

    bleus = []
    max_N = 15
    for n in xrange(1, max_N + 1):
        ngram = array([0.0] * 10)
        for pdata in data:
            hyp = pdata[: n][-1]
            ngram += hyp.ngram
        bleus.append(BLEU(ngram, False))      
        print "BLEU of the %d-th position: %f" %(n, bleus[-1])

    total_bleu1 = sum([2 ** (b + 1) / log(p + 2.0) for p, b in enumerate(bleus)])
    total_bleu2 = sum([b / log(p + 2.0) for p, b in enumerate(bleus)])
    print "method1: BLEU of top-%d: %f" %(max_N, total_bleu1)
    print "method2: BLEU of top-%d: %f" %(max_N, total_bleu2)

