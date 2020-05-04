#!/usr/bin/python

import sys

def extract(ln):
    p1 = ln.find(">") + 1
    p2 = ln.rfind("<")
    return [ln[: p1], ln[p1: p2], ln[p2: ]]

if __name__ == "__main__":
    assert len(sys.argv) == 2, "cmd fe"

    fou = file(sys.argv[1] + ".head", "w")
    for ln in file(sys.argv[1], "rU"):
        ln = ln.strip()
        if ln.startswith("<seg"):
            head, body, tail = extract(ln)
            print >> fou, head, "<s> " + body + " </s>", tail
        else:
            print >> fou, ln
    fou.close()            
    print "OK"            
            
