#!/usr/bin/env python2.7
  
from sys import argv
from time import time
from cvxopt.modeling import op, variable
from cvxopt import solvers, matrix

def analyze(ln):
    toks = ln.split()
    return matrix([float(w) for w in toks[: -1]]).T, float(toks[-1])

def test_linear(fe):
    fi = open(fe, "rU")
    r = []
    while True:
        try:
            h1, s1 = analyze(fi.next())
            h2, s2 = analyze(fi.next())
        except:
            break
        r.append(h2 - h1)
        if s1 < s2:
            r[-1] *= -1
    A = matrix(r)           
    C = matrix([1.0] * A.size[0])
    x = variable(A.size[1])
    f = matrix([0.0] * A.size[1]).T
    prob = op(f * x, [A * x + C <= 0]) 
    prob.solve(solver = "glpk")
    if prob.status == "optimal":
        return True, x.value
    else:
        return False, None

if __name__ == "__main__":
    for fe in argv[1:]:
        obj, ws = test_linear(fe)
        fo = open(fe + ".lp", "w")
        print >>  fo, " ".join(map(str, list(ws)))
        fo.close()
        print fe, obj
