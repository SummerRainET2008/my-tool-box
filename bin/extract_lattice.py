#!/usr/bin/python
'''
extract some src from a lattice file defined as follows:
<seg id=..>
3
src1
src2
src3
</seg>
'''

import sys

if __name__ == "__main__":
    assert len(sys.argv) == 3, "cmd infile n=(1,2,3)"

    lines = [ln.strip() for ln in file(sys.argv[1], "rU").readlines()]
    n = int(sys.argv[2])
    fw = file("%s.out%d" %(sys.argv[1], n), "w")
    i = 0
    while i < len(lines):
        if lines[i].startswith("<seg"):
            print >> fw, lines[i], lines[i + 1 + n], "</seg>"
            i += 6
        else:
            print >> fw, lines[i]
            i += 1
    print "over"


