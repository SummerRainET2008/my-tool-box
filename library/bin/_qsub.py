#!/usr/bin/python

from sys import argv
from os import getcwd, system
from random import randint
from optparse import OptionParser

cmd = '''
#PBS -N {name} 
#PBS -l nodes=1:ppn={thread}
#PBS -d %s

time {cmd} > log.%d 
''' 

if __name__ == "__main__":
    parser = OptionParser("cmd [options] your-commands")
    #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose", default = False, help = "don't print status messages to stdout")
    parser.add_option("-n", "--name", dest = "name", default = "Summer-main1.py", help = "showing name in the queue, default: Summer-main1.py")
    parser.add_option("-t", "--thread-num", dest = "thread_num", type = "int", default = "1", help = "showing name in the queue, default: 1")
    (options, args) = parser.parse_args()

    qid = randint(0, 100 * 10000)
    cmd = cmd % (getcwd(), qid)
    cmd = cmd.replace("{cmd}", " ".join(args))
    cmd = cmd.replace("{name}", options.name)
    cmd = cmd.replace("{thread}", str(options.thread_num))
    fe = "script.%d.pb" %qid 
    print >> file(fe, "w"), cmd
    #system("qsub %s" %fe)
