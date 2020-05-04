#!/usr/bin/env python

'''Version 1.
   1. Searching fold%d/systems/log_file
    
'''

from algorithm import *

def get_result(fn, tree_num):
    '''
    0:     evaluation: 999
    1:     train: 1:0.736192 2:0.728624 3:0.731031 4:0.738811 5:0.748271 6:0.757713 7:0.766762 8:0.775065 9:0.782236 10:0.788336 
    2:     vali:  1:0.701126 2:0.698173 3:0.705613 4:0.715807 5:0.726913 6:0.737462 7:0.748193 8:0.757182 9:0.765811 10:0.772392 
    3:     test:  1:0.713139 2:0.710357 3:0.716847 4:0.725687 5:0.735987 6:0.745962 7:0.755067 8:0.764199 9:0.772174 10:0.779214 
    4:
    5:     until now, best ndcgs
    6:     best train: 1:0.728464 2:0.725375 3:0.730997 4:0.738829 5:0.747495 6:0.757701 7:0.766658 8:0.774285 9:0.782263 10:0.787812 
    7:     best vali:  1:0.706368 2:0.698993 3:0.705923 4:0.716037 5:0.727319 6:0.737713 7:0.74841 8:0.757354 9:0.765974 10:0.772645 
    8:     best test:  1:0.7125 2:0.709784 3:0.71672 4:0.725673 5:0.735909 6:0.746038 7:0.754931 8:0.763781 9:0.7721 10:0.778955 
    9:
    10:    ERR: 0.463027           0.451628                0.458662
    11:    until now, best ERR             0.46168         0.452115                0.458771
    '''
    '''return NDCG@(1, 3, 10), ERR
    '''

    results = []
    reg     = re.compile(''':([\d.]+)''')

    for ln in open(fn):
        if ln.startswith("test:"):
            #print map(float, reg.findall(ln))
            results.append(list(itemgetter(0, 2, 9)(list(map(float, reg.findall(ln))))))
        elif ln.startswith("ERR:"):
            results[-1].append(float(ln.split()[-1]))
    if len(results) >= tree_num:
        return array(results[tree_num - 1])
    else:
        assert False, "%d < %d" %(len(results), tree_num)

def analysis(sys_name, fn_log, tree_num):
    n_fold  = len([fold for fold in listdir(".") if fold.startswith("fold")])
    results = [get_result("fold%d/%s/%s" %(fold + 1, sys_name, fn_log), tree_num) for fold in range(n_fold)]
    return results
    
if __name__ == "__main__":
    parser = OptionParser(usage = "cmd [optons] sys1 sys2 sys3...]")
    parser.add_option("-l", "--log",   dest = "fn_log",   default = None)
    parser.add_option("-t", "--tree",  dest = "tree_num", default = 1000, type = "int")
    #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose", default = False, help = "don't print status messages to stdout")
    (options, args) = parser.parse_args()

    for sys_name in args:
        print("System:", sys_name)
        print("%12s %12s %12s %12s %12s" %("fold", "NDCG@1", "NDCG@3", "NDCG@10", "ERR"))
        result = analysis(sys_name, options.fn_log, options.tree_num)
        for fold, (n1, n3, n10, err) in enumerate(result):
            print("%d\t%f\t%f\t%f\t%f" %(fold + 1, n1, n3, n10, err))
        print() 
        n1, n3, n10, err = sum(result) / len(result)
        print("%s\t%f\t%f\t%f\t%f" %("AVG", n1, n3, n10, err))
        print() 
