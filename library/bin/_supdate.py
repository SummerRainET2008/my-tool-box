#!/usr/bin/env python3
#coding: utf8

from library.bin import _scp
from palframe import nlp
from palframe.nlp import *
import optparse

if __name__ == "__main__":
  parser = optparse.OptionParser(usage = "cmd srcDir targetDir")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  parser.add_option("--exclude", dest="excludePattern", default=None)
  parser.add_option("-d", action = "store_true", dest = "delete",
                    default = False)
  (options, args) = parser.parse_args()
  assert len(args) == 2 and ("." == args[0] or "." == args[1])

  Logger.set_level(0)
  deleteOpt = "--delete" if options.delete else ""
  if options.excludePattern is not None:
    excludeOpt = f"--exclude={options.excludePattern}"
  else:   
    excludeOpt = ""

  srcDir, port1 = _scp.replace_server(args[0])
  tgtDir, port2 = _scp.replace_server(args[1])
  srcDir += "/"
  tgtDir += "/"

  if not nlp.is_none_or_empty(port1):
    port_opt = f"--port={port1}"
  elif not nlp.is_none_or_empty(port2):
    port_opt = f"--port={port2}"
  else:
    port_opt = ""

  cmd = f"rsync -ravutzhlog --progress -e ssh {port_opt} " \
        f"{srcDir} {tgtDir} {excludeOpt} {deleteOpt}"
  nlp.execute_cmd(cmd)

