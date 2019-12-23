#!/usr/bin/env python3
#coding: utf8

from pa_nlp import nlp
import optparse
import _scp

if __name__ == "__main__":
  parser = optparse.OptionParser(usage = "cmd srcDir targetDir")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  parser.add_option("--exclude", dest="excludePattern", default=None)
  parser.add_option("-d", action = "store_true", dest = "delete",
                    default = False)
  (options, args) = parser.parse_args()
  assert len(args) == 2 and ("." == args[0] or "." == args[1])
  
  deleteOpt = "--delete" if options.delete else ""
  if options.excludePattern is not None:
    excludeOpt = f"--exclude={options.excludePattern}"
  else:   
    excludeOpt = "" 
 
  srcDir = _scp.replaceServer(args[0]) + "/"
  tgtDir = _scp.replaceServer(args[1]) + "/"
  
  cmd = f"rsync -ravutzhlog --progress -e ssh " \
        f"{srcDir} {tgtDir}  {excludeOpt} {deleteOpt}"
  print(cmd)
  nlp.execute_cmd(cmd)
