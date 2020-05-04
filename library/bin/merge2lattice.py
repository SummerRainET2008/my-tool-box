#!/usr/bin/python

import sys

def extract(line):
    p1 = line.find(">")
    p2 = line.rfind("<")
    return [line[: p1 + 1], line[p1 + 1: p2], line[p2: ]]

def get_src(infe):
    for line in file(infe, "rU"):
        if line.startswith("<seg"):
            t = extract(line)
            yield t[1]

if __name__ == "__main__":
    assert len(sys.argv) > 1, "cmd, file1, file2"
    infes = sys.argv[1: ]
    finbase = file(infes[0], "rU")
    print "base:", infes[0]
    finother = infes[1: ]
    print "another:", finother
    fout = file("lattice.in", "w")
    srcite = [get_src(fe) for fe in finother]

    for line in finbase:
        if line.startswith("<seg"):
            t = extract(line)
            #print "line:", line
            #print "head:", t[0]
            #print "con:", t[1]
            #print "tail:", t[2]
            print >> fout, t[0]
            print >> fout, len(infes)
            print >> fout, t[1]
            for ite in srcite:
                print >> fout, ite.next()
            print >> fout, t[2]
        else:
            print >> fout, line,
    print "all is OK"
