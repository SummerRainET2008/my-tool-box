#!/usr/bin/env python3
#coding: utf8

from server_manager import *
from pa_nlp import nlp
import optparse

debug = False

def loadServerConfig():
  serverManager = ServerManager.get_instance()
  ret = {}
  for name in serverManager.get_server_name():
    ret[name] = serverManager.get_login(name)

  return ret

def replaceServer(addr):
  global servers
  if "@" not in addr:
    return addr

  servers = loadServerConfig()
  server = addr[: addr.index("@")]
  if server in servers:
    return addr.replace(server + "@", servers[server] + ":")
  return addr

def showServers():
  servers = loadServerConfig()
  for name in sorted(servers.keys()):
    print(name, servers[name])

if __name__ == "__main__":
  parser = optparse.OptionParser(usage = "cmd dev1@dir1 dir2")
  parser.add_option("--debug", action="store_true")
  parser.add_option("--recursive", action="store_true")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  (options, args) = parser.parse_args()

  if len(args) == 0:
    showServers()
    exit(0)

  debug = options.debug
  assert len(args) == 2

  srcDir, tgtDir = replaceServer(args[0]), replaceServer(args[1])
  dirOpt = "-r" if srcDir.endswith("/") else ""
  if options.recursive:
    cmd = "scp -r -oStrictHostKeyChecking=no %s %s %s" %(dirOpt, srcDir, tgtDir)
  else:
    cmd = "scp %s -oStrictHostKeyChecking=no %s %s" %(dirOpt, srcDir, tgtDir)
  print(cmd)
  nlp.execute_cmd(cmd)

