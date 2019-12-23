#!/usr/bin/python

from sys import argv

def get_weights(lamb):
    weights = [float(f) for f in file(lamb, "rU").readlines()[1].split()]
    print weights
    return weights

def compute_best(weigths, hyps):
    cands = []
    for hyp_id, hyp in enumerate(hyps):
        score = sum([float(f) * weights[id] for id, f in enumerate(hyp[1].split())])
        cands.append((score, " ||| ".join(hyp), hyp_id))
    opth = max(cands)
    return [opth[2], opth[1], opth[0]]
   
def get_nbest(fe):
    fin = file(fe, "rU")
    src_num = int(fin.next())
    yield src_num
    
    for src_id in xrange(src_num):
        fin.next()
        hyp_num = int(fin.next()) - 1
        #oracle_id = int(fin.next().split("|||")[-1])
        oracle_id = -1
        fin.next()
        hyps = [fin.next().strip().split("|||") for i in xrange(hyp_num)]
        yield [oracle_id, hyps]
        
if __name__ == "__main__":
    assert len(argv) == 3, "cmd lambda combined_oracle"
    
    fou = file(argv[2] + ".adjust", "w")
    weights = get_weights(argv[1])
    nbest_iter = get_nbest(argv[2])
    src_num = nbest_iter.next()
    crt_num = 0
    print >> fou, src_num
    for src_id in xrange(src_num):
        print src_id
        print >> fou, "src_id:", src_id
        oracle_id, hyps = nbest_iter.next()
        print >> fou, len(hyps) + 1
        [id, best_hyp, score] = compute_best(weights, hyps)
        if id == oracle_id:
            crt_num += 1
        print >> fou, best_hyp, "|||", "righ=%d, but=%d" %(oracle_id, id), "|||", score
        for hyp_id, hyp in enumerate(hyps):
            print >> fou, " ||| ".join(hyp)
    print "classifier accuracy:", crt_num / float(src_num)            
    fou.close()
            
    
