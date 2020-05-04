#!/usr/bin/env python
#coding: utf-8
 
from sys import stdout
from os import system, path
from time import time
from math import exp, log
from optparse import OptionParser
from re import compile
from operator import itemgetter
import pylab  

reg1, reg2 = compile("[\d.]+"), compile("[\d]+:([\d.]+)")

def analyze(log):
    fin = open(log)
    istest = False 
    data = []
    for ln in fin:
        if ln.startswith("evaluation"):
            values = list(map(float, reg1.findall(ln)))[1:]
            istest = True
        elif istest:
            buffs = list(map(float, reg2.findall(ln)))
            values.extend(buffs)
            data.append(values)
            istest = False
    return data 

if __name__ == "__main__":
    parser = OptionParser(usage = "cmd [optons] log...]")
    #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose", default = False, help = "don't print status messages to stdout")
    parser.add_option("-t", "--train", action = "store_true", dest = "train", default = False, help = "show train")
    parser.add_option("-v", "--vali",  action = "store_true", dest = "vali",  default = False, help = "show vali") 
    parser.add_option("-T", "--test",  action = "store_true", dest = "test",  default = False, help = "show test") 
    parser.add_option("-o", "--order", type = "int",          dest = "order", default = 1,     help = "show test") 
    (options, args) = parser.parse_args()

    data = analyze(args[0])
    print(data[0])

    col = lambda data, idx: [p[idx] for p in data]
    for log in args:
        data = analyze(log)
        if options.train:
            pylab.plot(list(map(itemgetter(0), data)), "+-", label = "%s: train" %log) 
        if options.vali:    
            pylab.plot(list(map(itemgetter(1), data)), "o-", label = "%s: vali"  %log) 
        if options.test:    
            pylab.plot(list(map(itemgetter(2 + options.order - 1), data)), "*-", label = "%s: test"  %log) 
        pylab.legend()
        pylab.grid()
    pylab.show()
