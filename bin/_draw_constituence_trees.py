#!/usr/bin/env python

from algorithm import *

if __name__ == "__main__":
    parser = OptionParser(usage = "cmd [optons] ..]")
    #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose", default = False, help = "don't print status messages to stdout")
    parser.add_option("-i", "--input", dest = "fn_tree", default = None)
    (options, args) = parser.parse_args()

    import nltk
    tree = open(options.fn_tree).read()
    nltk.Tree(tree).draw()
         
