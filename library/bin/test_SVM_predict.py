#!/home3/ly/xiatian/bin/python2.5

'''
    To transform the prediction from SVM to a formal nbest, so as to be computed the BLEU.
'''

from sys import argv
from math import log, exp
from operator import mul
from copy import deepcopy
from compute_combined_oracle import get_nbest

min_value = None
max_value = None

def get_predict(fe, src_num):
    predict = [int(float(p) + 0.5) for p in file(fe, "rU").read().split()]
    assert len(predict) == src_num
    for i in xrange(src_num):
        if predict[i] < min_value:
            predict[i] = min_value
        elif predict[i] > max_value:
            predict[i] = max_value
        yield predict[i]                    

if __name__ == "__main__":
    '''try:
        import psyco
        psyco.full()
        print "optimization on"
    except:
        print "optimization off"'''

    assert len(argv) == 6, "cmd SVM.predict combined_oracle nbest.out [min, max]"

    min_value = int(argv[4])
    max_value = int(argv[5])

    nbests = list(get_nbest(argv[2]))
    src_num = len(nbests)
    print "src_num:", src_num
    solus = list(get_predict(argv[1], src_num))
    fou = file(argv[3], "w")

    correct_num = 0
    print >> fou, src_num
    for src_id in xrange(src_num):
        correct = int(nbests[src_id][0].split("|||")[-1])
        nbest = nbests[src_id][1:]
        predict = solus[src_id]
        if predict == correct:
            correct_num += 1
        print >> fou, "src:", src_id
        print >> fou, 1
        print >> fou, nbest[predict]
    fou.close()   
    print "all is OK, precision = %f" %(correct_num / float(src_num))

