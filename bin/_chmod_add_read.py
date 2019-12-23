#!/usr/bin/env python3
#coding: utf8

from algorithm_3x import *

if __name__ == "__main__":
  os.system("clear")

  parser = optparse.OptionParser(usage = "cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  (options, args) = parser.parse_args()

  cwd = os.getcwd()

  cmd = "find . -type d -exec chmod -v a+x {} \;"
  executeCmd(cmd)

  cmd = "chmod -Rv a+r ."
  executeCmd(cmd)

