#!/usr/bin/env python3
#coding: utf8

from bin._scp import *
from pa_nlp import nlp

if __name__ == "__main__":
  parser = optparse.OptionParser(usage = "cmd server 'cmd'")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  (options, args) = parser.parse_args()

  if len(args) == 0:
    showServers()
    exit(0)

  if args == ["lab"]:
    nlp.execute_cmd("ssh -p 39002 summer@ml.pingan-labs.us")
    exit(0)

  loginServer = args[0] 
  if "@" not in loginServer:
    servers = loadServerConfig()
    loginServer = servers.get(loginServer, loginServer)
  
  cmd = "ssh -oStrictHostKeyChecking=no %s '%s'" %(loginServer, "" if len(args) == 1 else args[1])
  print(cmd)
  nlp.execute_cmd(cmd)

