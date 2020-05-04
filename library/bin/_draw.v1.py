#!/usr/bin/env python
#coding: utf-8
 
from sys import stdout
from os import system, path
from time import time
from math import exp, log
from optparse import OptionParser
from re import compile
import pylab  

reg = compile("[\d.]+")

def analyze(log):
    lns = [map(float, reg.findall(ln)) for ln in open(log) if ln.startswith("evaluation")]
    return lns

if __name__ == "__main__":
    parser = OptionParser(usage = "cmd [optons] log...]")
    #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose", default = False, help = "don't print status messages to stdout")
    parser.add_option("-t", "--train", action = "store_true", dest = "train", default = False, help = "show train")
    parser.add_option("-v", "--vali",  action = "store_true", dest = "vali",  default = False, help = "show vali") 
    parser.add_option("-T", "--test",  action = "store_true", dest = "test",  default = False, help = "show test") 
    (options, args) = parser.parse_args()

    col = lambda data, idx: [p[idx] for p in data]
    for log in args:
        data = analyze(log)
        if options.train:
            pylab.plot(col(data, 1), "+-", label = "%s: train" %log) 
        if options.vali:    
            pylab.plot(col(data, 2), "o-", label = "%s: vali"  %log) 
        if options.test:    
            pylab.plot(col(data, 3), "*-", label = "%s: test"  %log) 
        pylab.legend()
        pylab.grid()
    pylab.show()
