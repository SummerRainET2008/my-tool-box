#!/usr/bin/env python3
#coding: utf8

from library.include.server_manager import *
from palframe import nlp
from palframe.nlp import Logger
import optparse

def replace_server(addr):
  if "@" not in addr:
    return addr, None

  server_manager = ServerManager.get_instance()
  server_name = addr[: addr.index("@")]
  login_name = server_manager.get_login(server_name)
  port = server_manager.get_port(server_name)

  if login_name is not None:
    return addr.replace(server_name + "@", login_name + ":"), port

  return addr, port

def main():
  parser = optparse.OptionParser(usage = "cmd dev1@dir1 dir2")
  parser.add_option("--recursive", action="store_true")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  (options, args) = parser.parse_args()

  if len(args) == 0:
    exit(0)

  assert len(args) == 2

  src_dir, port1 = replace_server(args[0])
  tgt_dir, port2 = replace_server(args[1])
  cmds = ["scp -oStrictHostKeyChecking=no"]

  if not nlp.is_none_or_empty(port1):
    cmds.append(f"-P {port1}")
  elif not nlp.is_none_or_empty(port2):
    cmds.append(f"-P {port2}")

  if options.recursive:
    cmds.append("-r")

  cmds.append(f"{src_dir} {tgt_dir}")

  nlp.execute_cmd(*cmds)

if __name__ == "__main__":
  main()

