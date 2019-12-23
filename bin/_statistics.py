#!/usr/bin/env python
#coding: utf-8
 
from sys import stdout
from os import system, path
from time import time
from math import exp, log
from optparse import OptionParser
from re import compile

myprec = lambda h: [str(round(e, 6)) for e in h]

def compute_reversed_number(ds):
    if len(ds) <= 1: 
        return 0
    else:
        he = ds[0]
        s1 = filter(lambda e: e < he, ds[1:])
        s2 = filter(lambda e: e > he, ds[1:])
        return len(s1) + compute_reversed_number(s1) + compute_reversed_number(s2)

def extract_max(nums):
    b1 = max(nums, key = lambda e: e[0])
    b2 = max(nums, key = lambda e: e[1])
    b3 = max(nums, key = lambda e: e[2])
    return [myprec(b1), myprec(b2), myprec(b3)]

def extract(logfile):
    reg = compile('''[\d\.]+''')
    numbers = ["-".join(reg.findall(ln)[1:]) for ln in open(logfile) if ln.startswith("evaluation")]
    numbers = [map(float, e.split("-")) for e in numbers]
    numbers = [map(lambda v: v * 100, e) for e in numbers]
    return sorted(numbers)

def standard_deviation(samples):
    avg = sum(samples) / len(samples)
    d = sum([(e - avg) ** 2 for e in samples])
    return [avg, (d / len(samples)) ** 0.5]

if __name__ == "__main__":
    parser = OptionParser(usage = "cmd [optons] log.1 ...")
    #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose", default = False, help = "don't print status messages to stdout")
    (options, args) = parser.parse_args()
  
    numberss = []
    samples1, samples2 = [], []
    for fname in args:
        numbers = extract(fname)
        size = len(numbers)
        numberss.append(numbers)
        maxs = extract_max(numbers)
        samples1.append(float(maxs[0][2]))
        samples2.append(float(maxs[1][2]))
        #rnum = compute_reversed_number([e[2] for e in numbers])
        
        print "in file:", fname
        print "unique pair numbers:", size 
        print "choose best train performace:", maxs[0]
        print "choose best vali  performace:", maxs[1]
        print "choose best test  performace:", maxs[2]
        #print "error ranks", float(rnum) / size / (size - 1)
        print

    print "total"
    numberss = sum(numberss, [])
    size = len(numberss)
    maxs = extract_max(numberss)
    v1, d1 = standard_deviation(samples1)
    v2, d2 = standard_deviation(samples2)
    #rnum = compute_reversed_number([e[2] for e in numberss])
    #numberss.sort()
    print "unique pair numbers:", size 
    print "choose best train performace:", maxs[0], "standard: %f +- %f" %(v1, d1)
    print "choose best vali  performace:", maxs[1], "standard: %f +- %f" %(v2, d2)
    print "choose best test  performace:", maxs[2]
    #print "error ranks", float(rnum) / size / (size - 1)

    
