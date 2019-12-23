#!/usr/bin/env python

from algorithm import *
import nltk

def generate_tree(words, ldeps, rdeps, node_id):
    print("node_id:", node_id)
    w = words[node_id]
    lleaves, rleaves = "", ""
    for lw in sorted(ldeps[node_id]):
        lleaves += generate_tree(words, ldeps, rdeps, lw) 
    #lleaves = "( %s )" %lleaves if lleaves != "" else ""
    for rw in sorted(rdeps[node_id]):
        rleaves += generate_tree(words, ldeps, rdeps, rw) 
    #rleaves = "( %s )" %rleaves if rleaves != "" else "" 
    return "( %s %s %s )" %(w, lleaves, rleaves)

if __name__ == "__main__":
    parser = OptionParser(usage = "cmd [optons] file-name]")
    #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose", default = False, help = "don't print status messages to stdout")
    (options, args) = parser.parse_args()
    
    lns = '''poss(dog-2, My-1)
    nsubj(likes-4, dog-2)
    advmod(likes-4, also-3)
    root(ROOT-0, likes-4)
    xcomp(likes-4, eating-5)
    dobj(eating-5, sausage-6)
    '''.split("\n")

    args = ["/Users/summer/bin/tree.dependency.txt"]

    words = {}
    ldeps = defaultdict(list)
    rdeps = defaultdict(list)
    reg1  = re.compile(r"(.*?)\((.*?), (.*?)\)", re.DOTALL)
    reg2  = re.compile(r"(.*)-(.*)", re.DOTALL)
    for ln in open(args[0]):
    #for ln in lns:
        results = reg1.findall(ln)
        if len(results) == 0:
            continue
        rel, t1, t2 = results[0]
        w1, p1      = reg2.findall(t1)[0]
        w2, p2      = reg2.findall(t2)[0]
        p1, p2      = int(p1) - 1, int(p2) - 1
        words[p1]   = t1
        words[p2]   = t2
        print(p1, w1, p2, w2)
        if p1 > p2:
            ldeps[p1].append(p2)
        else:
            rdeps[p1].append(p2)
        print("left:", ldeps)
        print("right:", rdeps)
  
    tree = generate_tree(words, ldeps, rdeps, -1)
    print(tree)
    nltk.Tree(tree).draw()
        

