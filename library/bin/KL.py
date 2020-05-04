#!/usr/bin/python

from math import log
from sys import argv

def norm(v):
    sv = float(sum(v))
    return [f / sv for f in v]

def compute_KL(pprob, qprob):
    pprob = norm(pprob)
    qprob = norm(qprob)
    n = len(pprob)
    
    t = 0
    for i in xrange(n):
        t += -pprob[i] * log(qprob[i] / pprob[i]) 
    return t        

if __name__ == "__main__":
    assert len(argv) > 1, "cmd n p1 p2 ..pn q1 q2 ..qn"

    n = int(argv[1])
    pprob = [float(a) for a in argv[2: 2 + n]]
    qprob = [float(a) for a in argv[2 + n: 2 + n + n]]
    
    print compute_KL(pprob, qprob)

