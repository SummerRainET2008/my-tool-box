#!/usr/bin/python

import optparse
import sys

if __name__ == "__main__": 
    optparser = optparse.OptionParser()
    optparser.add_option("-i", dest = "ife", help = "input file")
    optparser.add_option("-o", dest = "ofe", help = "output file")
    optparser.add_option("-f", "--filter", type = "int", dest = "filter", help = "filter tree, 1 or 0")
    (opts, args) = optparser.parse_args() 

    assert opts.ife is not None
    assert opts.ofe is not None
    assert opts.filter is not None

    fw = file(opts.ofe, "w")
    for ind, line in enumerate(file(opts.ife, "rU")):
        line = line.rstrip()
        if line.startswith("<srcword"):
            line = line.replace("<srcword", "<tgtword")
            print >> fw, line
        elif line.startswith("<tgtword"):
            line = line.replace("<tgtword", "<srcword")
            print >> fw, line
        elif line.startswith("<wordalignment>"):
            f = line.find(">")
            t = line.rfind("<")
            line = line[f + 1: t]
            nalign = []
            for s in line.split():
                s = s[: -2].split(":")
                f, t = s[0], s[1]
                nalign.append("%s:%s/0" %(t, f))
            print >> fw, "<wordalignment>%s</wordalignment>" %(" ".join(nalign))
        elif opts.filter == 1:
            if line.startswith("<srctree") or line.startswith("<tgttree") or line.startswith("<tgtdeptree"):
                continue
            else:
                print >> fw, line
        else:
            print >> fw, line
        if ind % 10000 == 0:
            print ind
    print ind
    print "OK"
    
