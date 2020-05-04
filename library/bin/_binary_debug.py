#!/usr/bin/env python

from algorithm import *

def get_input(msg):
    while True:
        ret = raw_input("%s, input y[yes] or n[o]: " %msg).split()
        if ret in ['y', 'n']:
            return ret

def binary_guess(f, t):
    if t - f + 1 == 2:
        print "get the bug in line %d to %d" %(f, t)
        return

    mid = (f + t) / 2 
    print "comment out line %d to %d, lines: %d" %(f, mid, mid - f + 1)
    c = get_input("Program has a bug")
    if c == 'y':
        binary_guess(mid + 1, t)
    elif c == 'n':
        print "locating the bug in line %d to %d, and uncomment out %d to %d" %(f, mid, f, mid)
        binary_guess(f, mid)

if __name__ == "__main__":
    parser = OptionParser(usage = "cmd [optons] ..]")
    #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose", default = False, help = "don't print status messages to stdout")
    parser.add_option("-f", dest = "line_f", type = "int") 
    parser.add_option("-t", dest = "line_t", type = "int")
    (options, args) = parser.parse_args()

    f, t = options.line_f, options.line_t
    binary_guess(f, t)
      
