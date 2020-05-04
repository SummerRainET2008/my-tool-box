#!/usr/bin/env python2.7
#coding: utf-8
  
from sys import argv
from time import time
from os import listdir, path, rename, getcwd, chdir
from optparse import OptionParser

def lower_filename(pattern, cdir, files):
    chdir(cdir)
    files = [fname for fname in files if fname.endswith(".pdf")]
    for fname in files:
        if not fname.startswith("["):
            print "warning: %s, not starting with [..]" %fname
            print
        if fname != fname.lower():  
            try:
                rename(fname, fname.lower())
                print "rename to %s" %(fname.lower())
            except:
                print "can't rename", fname
                print "current directory:", cdir
                print

if __name__ == "__main__":
    path.walk(getcwd(), lower_filename, None)


