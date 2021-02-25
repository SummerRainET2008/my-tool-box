#!/usr/bin/env python3
#coding: utf8

from palframe.nlp import Logger
from library.include.server_manager import *

if __name__ == "__main__":
  parser = optparse.OptionParser(usage = "cmd server 'cmd'")
  parser.add_option("--ping", action="store_true")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  (options, args) = parser.parse_args()

  server_manager = ServerManager.get_instance()
  if len(args) == 0:
    exit(0)

  login_server = args[0]
  if "@" in login_server:
    Logger.error("You should use ssh instead.")
    exit(0)

  if options.ping:
    ip = server_manager.get_ip(login_server)
    nlp.execute_cmd(f"ping {ip}")
    exit(0)

  cmds = [f"ssh -oStrictHostKeyChecking=no"]
  cmds.append(f"{server_manager.get_login(login_server)}")
  port = server_manager.get_port(login_server)
  if port != "":
    cmds.append(f"-p {port}")

  nlp.execute_cmd(*cmds)

