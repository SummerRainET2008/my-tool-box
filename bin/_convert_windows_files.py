#!/usr/bin/env python3

import optparse

if __name__ == "__main__":
  parser = optparse.OptionParser(usage = "cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose", \
      #default = False, help = "don't print status messages to stdout")
  (options, args) = parser.parse_args()

  for fn in args:
    try:
      txt = open(fn).read()
      txt = "".join([ch for ch in txt if ord(ch) != 13])
      print(txt, file=open(fn + ".converted", "w"))
      print(fn, "is OK")
    except:
      break

