#!/home3/ly/xiatian/Install/Python-2.5.2/python

from sys import argv

def get_nbest(fe):
    fin = file(fe, "rU")
    src_num = int(fin.next())
    for src_id in xrange(src_num):
        fin.next()
        hyp_num = int(fin.next()) - 1
        orcID = int(fin.next().split("|||")[-1])
        hyps = [fin.next().split("|||")[1] for id in xrange(hyp_num)]
        yield [orcID, hyps]

def shift(hyps):
    hyps = [hyp.split() for hyp in hyps]
    rn, cn = len(hyps), len(hyps[0])
    rhyps = [[None for i in xrange(rn)] for j in xrange(cn)]
    for r in xrange(rn):
        for c in xrange(cn):
            rhyps[c][r] = hyps[r][c]
    return rhyps            

def gen_feat(ife, ofe):
    fou = file(ofe, "w")
    for orcID, hyps in get_nbest(ife):
        print >> fou, 1
        print >> fou, 4, orcID
        print >> fou

        feats = shift(hyps)
        for feat_id, feat in enumerate(feats):
            print >> fou, feat_id
            print >> fou, "1 0"
            print >> fou, len(feat)
            print >> fou, "\n".join(["%d %s" %(i, f) for i, f in enumerate(feat)])
            print >> fou
        print >> fou, -1
        print >> fou
    fou.close()        

if __name__ == "__main__":
    for fe in argv[1:]:
        gen_feat(fe, fe + ".crf_feat")
        print fe, "is OK"
            
