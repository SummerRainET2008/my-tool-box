#!/usr/bin/env python
#coding: utf-8

'''Version 2
   1. Only applicable for such format.
   evaluation: 999
   train: 1:0.533184 2:0.510374 3:0.5023 4:0.500378 5:0.499823 6:0.501302 7:0.503441 8:0.505095 9:0.507519 10:0.510113 
   vali:  1:0.478619 2:0.469255 3:0.468198 4:0.471322 5:0.474835 6:0.478499 7:0.481286 8:0.484836 9:0.487556 10:0.492072 
   test:  1:0.473729 2:0.471476 3:0.466155 4:0.468954 5:0.471675 6:0.475426 7:0.477224 8:0.481267 9:0.484443 10:0.487584 

   until now, best ndcgs
   best train: 1:0.524271 2:0.49466 3:0.500951 4:0.498669 5:0.498279 6:0.499978 7:0.501922 8:0.499288 9:0.507226 10:0.510113 
   best vali:  1:0.486981 2:0.472652 3:0.468995 4:0.473229 5:0.475695 6:0.478709 7:0.481915 8:0.485594 9:0.487984 10:0.492072 
   best test:  1:0.46839 2:0.468317 3:0.465569 4:0.46783 5:0.470813 6:0.474481 7:0.476442 8:0.478739 9:0.484221 10:0.487584 

   ERR: 0.390993           0.372679                0.36488

   2. get_style() is a very useful function.

'''

from algorithm import *
from re import compile
import pylab  

reg = compile(":([\d.]+)")

def extract(ln):
    toks = reg.findall(ln)
    assert len(toks) == 10, ln
    return list(map(float, toks))

def analyze(log):
    train_measures, vali_measures, test_measures = [], [], []     
    for ln in open(log):
        if ln.strip().startswith("train:"):
            train_measures.append(extract(ln))
        elif ln.strip().startswith("vali:"):    
            vali_measures.append(extract(ln))
        elif ln.strip().startswith("test:"):
            test_measures.append(extract(ln))
        elif ln.strip().startswith("ERR:"):
            toks = ln.split() 
            train_measures[-1].append(toks[1])
            vali_measures[-1].append(toks[2])
            test_measures[-1].append(toks[3])
    return train_measures, vali_measures, test_measures

def col(result, measure):
    measure = measure.lower()
    if measure.startswith("ndcg@"):
        order = int(measure[5:]) - 1
    elif measure.startswith("err"):
        order = 10
    else:
        assert False, measure
    return [ln[order] for ln in result]      

def get_style():
    cols   = "rb"
    lines  = "-", "--", "-.", ":", "-.", 
    #lines  = "-", "--", "-.", ":", "-.", "o-", "-+", "-^", "-v", "-*", "-x"
   
    ret = []
    for l, c in product(lines[0: 10], cols):
        app   = c + l
        ret.append(app)
        #yield app
    #shuffle(ret)
    for app in ret:
        print(app)
        yield app

    '''data = map(float, range(5)) * 10
    shuffle(data)
    data = array(data)
    print data
    num = 0
    for l, c in product(lines[0: 10], cols):
        app   = c + l
        data += 0.32
        print app
        pylab.plot(data, app, label = app)
        pylab.legend()
        pylab.grid()
        num += 1
        if num == 25:
            break
    pylab.show()
    exit(1)'''

#get_style()

if __name__ == "__main__":
    parser = OptionParser(usage = "cmd [optons] log...]")
    #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose", default = False, help = "don't print status messages to stdout")
    parser.add_option("-l", "--legend", action = "store_true", dest = "legend",     default = False, help = "show train")
    parser.add_option("-t", "--train",  action = "store_true", dest = "train",      default = False, help = "show train")
    parser.add_option("-v", "--vali",   action = "store_true", dest = "vali",       default = False, help = "show vali") 
    parser.add_option("-T", "--test",   action = "store_true", dest = "test",       default = False, help = "show test") 
    parser.add_option("-m", "--measure",                       dest = "measure",    default = "NDCG@10", help = "NDCG@1 .. NDCG@10 ERR")
    parser.add_option("-o", "--outfile",                       dest = "outfile",    default = None,      help = "save file as a pdf")
    (options, args) = parser.parse_args()

    style = get_style()
    for log in args:
        result1, result2, result3 = analyze(log)
        result1, result2, result3 = col(result1, options.measure), col(result2, options.measure), col(result3, options.measure),
        #print result[: 3]
        if options.train:
            pylab.plot(result1, next(style), label = "%s: train" %log) 
            #print "\n".join(map(str, result1))
        if options.vali:    
            pylab.plot(result2, next(style), label = "%s: vali"  %log) 
            #print "\n".join(map(str, result2))
        if options.test:    
            pylab.plot(result3, next(style), label = "%s: test"  %log) 
            #print "\n".join(map(str, result3))
        if options.legend:
            pylab.legend()
        print()
    pylab.grid()
    if options.outfile is not None:
        pylab.savefig(options.outfile)
    else:    
        pylab.show()    
