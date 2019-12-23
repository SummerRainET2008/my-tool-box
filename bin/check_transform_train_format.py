#!/usr/bin/python

"""
1. check the training data.
2. trainsform the urgly sgm to a flat txt.
3. every align would start with 0.

src
tgt
align
"""

import sys
import time

def get_rec(fe):
    fin = file(fe, "rU")
    src = tgt = align = None
    while True:
        try:
            line = fin.next().rstrip()
        except StopIteration:
            break
        if line.startswith("<srcword"):
            src = line
        elif line.startswith("<tgtword"):
            tgt = line
        elif line.startswith("<wordalign"):
            align = line
            yield [src, tgt, align]
            src, tgt, align = None, None, None

def extract(s):
    return s[s.find(">") + 1: s.rfind("<")]

def valid(src, tgt, align):
    if src is None or tgt is None or align is None:
        return None
    src = extract(src)
    tgt = extract(tgt)
    align = extract(align)
    
    nsrc = src.count("/@")
    ntgt = tgt.count("/@")
    src = src.replace("/@", " ")
    tgt = tgt.replace("/@", " ")
    align = align.replace("/0", " ")
    na = []
    for ta in align.split():
        tmp = ta.split(":")
        f, t = float(tmp[0]) - 1, float(tmp[1]) - 1
        if not 0 <= f < nsrc or not 0 <= t < ntgt:
            return None
        na.append("%d-%d" %(f, t))
    return [" ".join(src.split()), " ".join(tgt.split()), "  ".join(na)]

if __name__ == "__main__":
    assert len(sys.argv) == 2, "cmd train"

    fou = file(sys.argv[-1] + ".flat", "w")
    
    error_no = 0
    count = 0

    for node in get_rec(sys.argv[1]):
        count += 1
        if count % 10000 == 0:
            print count
        nnode = valid(node[0], node[1], node[2])
        if nnode is None:
            error_no += 1
            print "error ID:", error_no
            print "src:", node[0] 
            print "tgt:", node[1]
            print "align:", node[2]
            print 
        else:            
            print >> fou, nnode[0]
            print >> fou, nnode[1]
            print >> fou, nnode[2]
            print >> fou
    print "OK"                
