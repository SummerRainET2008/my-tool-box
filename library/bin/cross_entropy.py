#!/usr/bin/python
# -*- coding: utf-8 -*-

from sys import argv
from math  import log10, e
import time

def compute1(nb_fe):
    fin = file(nb_fe, "rU")
    n = int(fin.next())
    lms = []
    for i in xrange(n):
        fin.next()
        m = int(fin.next()) 
        if m == 0:
            print "warning: nbest = 0"
        for j in xrange(m):
            line = fin.next()
            lm = float(line.split("|||")[1].split()[-1])
            if j == 0:
                lms.append(lm)
    print "src num = %d" %len(lms)                
    return -sum(lms) / len(lms)                
                
def compute2(nb_fe):
    fin = file(nb_fe, "rU")
    n = int(fin.next())
    lms = []
    num = 0
    for i in xrange(n):
        fin.next()
        m = int(fin.next()) 
        if m == 0:
            print "warning: nbest = 0"
        for j in xrange(m):
            line = fin.next()
            lm = float(line.split("|||")[1].split()[-1])
            if j == 0:
                lms.append(lm)
                num += len(line.split("|||")[0].split())
    print "src num = %d" %len(lms)                
    return -sum(lms) / num

def compute3(nb_fe):
    fin = file(nb_fe, "rU")
    n = int(fin.next())
    lms = []
    num = 0
    for i in xrange(n):
        fin.next()
        m = int(fin.next()) 
        if m == 0:
            print "warning: nbest = 0"
        for j in xrange(m):
            line = fin.next()
            lm = float(line.split("|||")[1].split()[-1]) * log10(e) 
            if j == 0:
                lms.append(lm)
                num += len(line.split("|||")[0].split())
    print "src num = %d" %len(lms)                
    return -sum(lms) / num

           
if __name__ == "__main__":
    print compute1(argv[1])
    print compute2(argv[1])
    print compute3(argv[1])
