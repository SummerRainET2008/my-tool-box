#!/usr/bin/python

import sys

def get_src_sent(fe):
    for ln in file(fe, "rU"):
        if ln.startswith("<seg"):
            yield ln[ln.find(">") + 1: ln.rfind("<")]

if __name__ == "__main__":
    assert len(sys.argv) == 3, "cmd nbest src"
    fin = file(sys.argv[1], "rU")
    fou = file(sys.argv[1] + ".com", "w")
    src_ite = get_src_sent(sys.argv[2])
    
    print >> fou, "<text>"
    
    n = int(fin.next())
    for i in xrange(n):
        print >> fou, "<sent No=%d>" %(i + 1)
        print >> fou, "<Chinese> %s </Chinese>" %(src_ite.next())
        print >> fou, "<nbest>"

        #print fin.next(),
        m = int(fin.next())
        for j in xrange(m):
            ln = fin.next().rstrip().split(" ||| ")
            print >> fou, "<hyp> %s </hyp>" %ln[0]
            print >> fou, "<sysid>chiero</sysid>"
            print >> fou, "<score>%s</score>" %ln[-1]
            print >> fou, "<alignment></alignment>"
            
        print >> fou, "</nbest>"
        print >> fou, "</sent>"

    print >> fou, "</text>"
            
    fin.close()
    fou.close()
    print "OK"        
        
