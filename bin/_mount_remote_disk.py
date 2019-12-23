#!/usr/bin/env python3
#coding: utf8

import optparse
import os

if __name__ == "__main__":
  parser = optparse.OptionParser(usage = "cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
  #default = False, help = "")
  parser.add_option(
    "--server_path",
    default="summer@192.168.1.201:/media/workspace/summer",
    help="account@IP:dir, no blanks permitted in dir, "
         "default summer@192.168.1.201:/media/workspace/summer"
  )
  parser.add_option(
    "--local_path",
    default = "/home/summer/summer201",
    help = "default '/home/summer/summer201'"
  )
  (options, args) = parser.parse_args()

  server_path = options.server_path
  local_path = options.local_path
  short_local_path = os.path.basename(local_path)
  os.system(
    f"sudo sshfs {server_path} {local_path} -o idmap=user -o allow_other"
  )

