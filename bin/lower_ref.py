#!/usr/bin/python

from sys import argv

fou = file(argv[1] + ".lower", "w")
num = 0
for ln in file(argv[1], "rU"):
    if ln.startswith("<seg"):
        print >> fou, ln.lower(),
        num += 1
    else:
        print >> fou, ln,
fou.close()
print num
            
