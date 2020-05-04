#!/usr/bin/python

from os import listdir, path, chdir, getcwd, system
from sys import argv
from copy import deepcopy as copy

def modify(fe):
    if " " not in fe:
        return fe
    else:
        nfe = "-".join(fe.split())
        user = raw_input("modify '%s/%s' --> '%s/%s' ? " %(getcwd(), fe, getcwd(), nfe))       
        #user = "yes"
        if user == "yes":
            system('''mv "%s/%s" %s/%s''' %(getcwd(), fe, getcwd(), nfe))
            return nfe
        else:
            return fe

def walk(dep):
    cwd = getcwd()
    for ind, fe in enumerate(listdir(".")):
        if path.isdir(fe) and not fe.startswith("."):
            fe = modify(fe)            
            print dep * 2 * " ", "entering", fe
            chdir(fe)
            walk(dep + 1)
            chdir(cwd)

if __name__ == "__main__":
    print listdir(".")
    walk(0)

