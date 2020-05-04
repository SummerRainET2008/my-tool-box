#!/usr/bin/env python3
#coding: utf8

from library.include.server_manager import *
from pa_nlp import nlp
import optparse

def replace_server(addr):
  global servers
  if "@" not in addr:
    return addr

  servers = ServerManager.get_instance()
  server = addr[: addr.index("@")]
  if server in servers:
    return addr.replace(server + "@", servers[server] + ":")
  return addr

if __name__ == "__main__":
  parser = optparse.OptionParser(usage = "cmd dev1@dir1 dir2")
  parser.add_option("--recursive", action="store_true")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  (options, args) = parser.parse_args()

  if len(args) == 0:
    exit(0)

  assert len(args) == 2

  srcDir, tgtDir = replace_server(args[0]), replace_server(args[1])
  dirOpt = "-r" if srcDir.endswith("/") else ""
  if options.recursive:
    cmd = "scp -r -oStrictHostKeyChecking=no %s %s %s" %(dirOpt, srcDir, tgtDir)
  else:
    cmd = "scp %s -oStrictHostKeyChecking=no %s %s" %(dirOpt, srcDir, tgtDir)
  print(cmd)
  nlp.execute_cmd(cmd)

