#!/usr/bin/python

__date__ = "2009/12/12"

from sys import argv

lfmt = '''\t<et>%s</et>,<et>%s</et>,%s,0'''       # ref-word, hyp-word, flag
fmt = '''\
<hyp refid=%d wrd_cnt=-1 num_errs=%s>
%s
</hyp>
'''

def gen_recode(id, ref, hyp, alin, sco):
    refiter, hypiter, aliniter = iter(ref), iter(hyp), iter(alin)
    r = []
    for a in alin:
        if a == " ": 
            r.append(lfmt %(refiter.next(), hypiter.next(), "C"))
        elif a in "YST":            # synomy, substitution, stemming.
            r.append(lfmt %(refiter.next(), hypiter.next(), "S"))
        elif a == "D":              # delete the reference word.
            r.append(lfmt %(refiter.next(), "", a))
        elif a == "I":              # insert a word in reference.
            r.append(lfmt %("", hypiter.next(), a))
        else:
            print "unknow alignment:", a
            exit(1)
    return fmt %(id, sco[0].strip(), "\n".join(r))            

if __name__ == "__main__":
    assert len(argv) == 3, "cmd terp.pra [dev|tst].xml"

    refs = [ln[ln.find(":") + 1:].split() for ln in file(argv[1], "rU") if ln.startswith("Reference:")]
    hyps = [ln[ln.find(":") + 1:].split() for ln in file(argv[1], "rU") if ln.startswith("Hypothesis After Shift:")]
    alns = [ln[ln.find("(") + 1: ln.rfind(")")] for ln in file(argv[1], "ru") if ln.startswith("Alignment:")]
    scos = [ln[ln.find("(") + 1: ln.rfind(")")].split("/") for ln in file(argv[1], "rU") if ln.startswith("Score:")]
    assert len(refs) == len(hyps) == len(alns) == len(scos)
    size = len(refs)

    fou = file(argv[2], "w")
    for i in xrange(size):
        print >> fou, gen_recode(i, refs[i], hyps[i], alns[i], scos[i])
    fou.close()
    print "all is OK"
    


