#!/usr/bin/python

from sys import argv
from os import system

if __name__ == "__main__":
    assert len(argv) == 3, "cmd isofile mount-place"
    cmd = '''sudo mount -t iso9660 -o loop "%s" "%s" ''' %(argv[1], argv[2])
    print "OK, it is mounted!" if system(cmd) == 0 else "failed"
            
