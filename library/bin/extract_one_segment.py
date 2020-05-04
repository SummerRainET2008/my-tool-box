#!/usr/bin/python

import sys

if __name__ == "__main__":
    for fe in sys.argv:
        fin = file(fe, "rU")
        fw = file(fe + ".new", "w")
        lines = [line.rstrip() for line in fin.readlines()]
        ind = 0
        while ind < len(lines):
            line = lines[ind]
            if line.startswith("<seg"):
                print >> fw, line, lines[ind + 2], "</seg>"
                ind += 6
            else:
                ind += 1
                print >> fw, line
        print fe, "OK"
