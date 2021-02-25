#!/usr/bin/env python3
#coding: utf8

from palframe import nlp
import optparse

if __name__ == "__main__":
  parser = optparse.OptionParser(usage = "cmd [optons] kw]")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  parser.add_option("-t", dest = "fn_ext",
                    default = "java,scala,c,cpp,hpp,h,py", 
                    help = "default java,scala,c,cpp,hpp,h,py")
  (options, args) = parser.parse_args()

  cmd_tpt = r"find . -iregex '.*\.%s' -exec grep -iHn '%s' {} \;"
  if len(args) > 0:
    for fn_ext in options.fn_ext.split(","):
      cmd = cmd_tpt %(fn_ext, args[0])  
      #print cmd
      nlp.execute_cmd(cmd)

