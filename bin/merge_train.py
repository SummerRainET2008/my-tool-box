#!/usr/bin/python

"""
merge three segmentations to a new traing file. 
"""

import sys
import time

def get_rec(fe):
    fin = file(fe, "rU")
    while True:
        try:
            src = fin.next().rstrip()
            tgt = fin.next().rstrip()
            align = fin.next().rstrip()
            fin.next()
        except StopIteration:
            break
        if src is not None and tgt is not None and align is not None:
            yield [src, tgt, align]
        src, tgt, align = None, None, None
            

if __name__ == "__main__":
    assert len(sys.argv) > 1, "cmd train1 train2 train3"

    fou = file("new_train.xml", "w")
    records_ite = [get_rec(fe) for fe in sys.argv[1: ]]
    rec_num = 0

    while True:
        _records_ite = []
        for rec_ite in records_ite:
            try:
                rec = rec_ite.next()
                _records_ite.append(rec_ite)
                print >> fou, rec[0]
                print >> fou, rec[1]
                print >> fou, rec[2]
                print >> fou
            except StopIteration:
                pass
        records_ite = _records_ite                            
        if len(records_ite) == 0:
            break
        rec_num += 1
        if rec_num % 10000 == 0:
            print "record:", rec_num
    fou.close()
    print "%s is OK" %__file__
            
