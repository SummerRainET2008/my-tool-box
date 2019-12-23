#!/usr/bin/env python

from algorithm import *

def run(cmd):
    print cmd
    if system(cmd) != 0:
        print "warning:", cmd

# formal run
if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-m", "--music", dest = "music_size", default = 1, type = "int")
    (options, args) = parser.parse_args()

    assert len(argv) in [6], "cmd src ref nbest del-tmp-file[1|0] case-sensitive[1|0]"
    
    cwd = getcwd()
    nbestfe = path.realpath(argv[3])
    srcfe = path.realpath(argv[1])
    reffe = path.realpath(argv[2])
    del_tmp = argv[4] == "1"
    case = argv[5] == "1"

    print "-" * 20
    print "current work-directory:", cwd 
    print "real path of nbest:", nbestfe
    print "real path of srcfe:", srcfe
    print "real path of reffe:", reffe
    print "-" * 20
    print

    fe1 = "%s.tran" %nbestfe
    fe2 = "%s.rst"  %nbestfe
    fe3 = "%s.bleu" %nbestfe

    chdir("/home3/jwb/xiatian/tools/TestNbestBleu")
        
    run("./newextract.py -i %s -o %s -s %s" %(nbestfe, fe1, srcfe))

    run("./post_deal -o %s -n %s -l" %(fe1, fe2))

    cont = file(fe2, "rU").read()
    cont = cont.replace("<s>", "").replace("</s>", "").replace("NULL", "")
    print >> file(fe2, "w"), cont

    if case:
        run("./mteval-v11b.pl -c -r %s -s %s -t %s | tee %s" %(reffe, srcfe, fe2, fe3))
    else:
        run("./mteval-v11b.pl    -r %s -s %s -t %s | tee %s" %(reffe, srcfe, fe2, fe3))

    if del_tmp:
        run("rm %s" %fe1)
        run("rm %s" %fe2)


    
