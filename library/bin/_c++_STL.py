#!/usr/bin/env python

from algorithm import *

if __name__ == "__main__":
  usage = '''
  cmd 
  or
  cmd algorithm 
  '''

  parser = optparse.OptionParser(usage = usage) 
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose", \
      #default = False, help = "don't print status messages to stdout")
  (options, args) = parser.parse_args()

  if args is None:
    cmd = "open http://www.cplusplus.com/reference/"
  else:   
    cmd = "open http://www.cplusplus.com/" + args[0]

  os.system(cmd)   


