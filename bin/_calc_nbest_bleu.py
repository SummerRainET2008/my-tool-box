#!/usr/bin/env python

import sys
if "/nfs/18/wsu0215/include" not in sys.path:
    sys.path.append("/nfs/18/wsu0215/include")
from algorithm import *

TOOL_PATH = "/nfs/18/wsu0215/install/test_nbest_bleu"

class Hyp:
    def __init__(self, ln):
        toks       = ln.split("|||")
        self.tran  = toks[0].strip()
        self.feats = array(map(float, toks[1].split()))
        self.score = 0 

    def update_score(self, weight):
        self.score = weight.dot(self.feats)

    @staticmethod
    def read_nbest(fn_nbest):
        fin = open(fn_nbest)
        src_size = int(fin.next())
        data     = []
        for sid in xrange(src_size):
            fin.next()
            hyp_size = int(fin.next())
            pdata    = [Hyp(fin.next()) for hi in xrange(hyp_size)]
            data.append(pdata)
        return data      
       
    @staticmethod
    def rerank(data, weight):
        print "reranking..."
        for pdata in data:
            map(methodcaller("update_score", weight), pdata)
            pdata.sort(key = lambda h: -h.score)

def run(cmd):
    print cmd
    if system(cmd) != 0:
        print "warning:", cmd

# formal run
if __name__ == "__main__":
    parser = OptionParser(usage = "[option] nbest1 nbest2...")
    #parser.add_option("-m", "--music", dest = "music_size", default = 1, type = "int")
    parser.add_option("-s", "--src",    dest = "fn_src")                 
    parser.add_option("-r", "--ref",    dest = "fn_ref")                 
    parser.add_option("-w",             dest = "fn_weight",     default = None, help = "Only applicable for the lambda file")
    parser.add_option("--tmp",          dest = "keep_tmp",      action  = "store_true")
    parser.add_option("--lower_case",   dest = "lower_case",    action  = "store_true")
    (options, args) = parser.parse_args()

    print "src file:", options.fn_src
    print "ref file:", options.fn_ref
    print "keep tmp:", options.keep_tmp
    print "cased   :", options.lower_case

    cwd = getcwd()
    for fn_nbest in args:
        rid = randint(0, 1024 * 1024)
        fe1 = "tmp.%d.%s.reranked" %(rid, fn_nbest)
        fe2 = "tmp.%d.%s.rst"  %(rid, fn_nbest)
        fe3 = "tmp.%d.%s.rst.post"  %(rid, fn_nbest)
        fe4 = "tmp.%d.%s.bleu" %(rid, fn_nbest)

        data = Hyp.read_nbest(fn_nbest)
        if options.fn_weight is not None:
            weight = array(map(float, open(options.fn_weight).readlines()[1].split()))
            Hyp.rerank(data, weight)
        
        fou = open(fe1, "w")
        print >> fou, len(data)
        for sid, pdata in enumerate(data):
            pdata = pdata[: 30]
            print >> fou, "srcid:", sid
            print >> fou, len(pdata) 
            for hyp in pdata:
                print >> fou, hyp.tran, "|||", " ".join(map(str, hyp.feats)), "|||", hyp.score
        fou.close()          

        run("%s/newextract.py -i %s -o %s -s %s" %(TOOL_PATH, fe1, fe2, options.fn_src))
        run("%s/post_deal -o %s -n %s -l" %(TOOL_PATH, fe2, fe3))

        cont = open(fe3).read().replace("<s>", "").replace("</s>", "").replace("NULL", "")
        print >> open(fe3, "w"), cont

        if not options.lower_case:
            run("%s/mteval-v11b.pl -c -r %s -s %s -t %s | tee %s" %(TOOL_PATH, options.fn_ref, options.fn_src, fe3, fe4))
        else:
            run("%s/mteval-v11b.pl    -r %s -s %s -t %s | tee %s" %(TOOL_PATH, options.fn_ref, options.fn_src, fe3, fe4))

        if not options.keep_tmp:
            for fname in [fe1, fe2, fe3, fe4]:
                run("rm %s" %fname)

