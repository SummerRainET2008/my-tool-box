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
    assert len(sys.argv) == 2, "cmd lattice_file"

    fw = file("%s.plain" %(sys.argv[1]), "w")
    f = file(sys.argv[1], "rU")
    while True:
        try:
            line = f.next()
            if line.startswith("<seg"):
                n = int(f.next())
                for i in xrange(n):
                    print >> fw, f.next(),
        except StopIteration:
            break            
    print "over"


