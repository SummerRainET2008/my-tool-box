#!/home3/jwb/xiatian/bin/python2.5

from sys import argv, path
from os import system
path.append("/home3/jwb/xiatian/include")
from mitel_algorithm import Nbest
from time import time

def inner_product(fs, ws):
    return sum([f * w for [f, w] in zip(fs, ws)])

def sort_nbest(nbest):
    lt = []
    for hyp in nbest:
        lt.append([-inner_product([float(f) for f in hyp[1].split()], WEIGHTS), hyp[0], hyp[1]])
    return sorted(lt) 

if __name__ == "__main__":
    assert len(argv) == 3, "cmd config.file nbest"
    execfile(argv[1])

    start = time()
    print "reading data..."                    
    nbests = Nbest().read_nbest(argv[2])
    print "sorting nbests..."
    fou = file(argv[2] + ".sort", "w")
    print >> fou, len(nbests)
    for src_id, nb in enumerate(nbests):
        print >> fou, "src_id", src_id
        print >> fou, len(nb)
        snb = sort_nbest(nb)
        print >> fou, "\n".join(["%s ||| %s ||| %f" %(h, f, -s) for s, h, f in snb])
    fou.close()   
    print "time: %f s" %(time() - start)
