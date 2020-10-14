#!/usr/bin/env python3

import sys
import platform
from pa_nlp import nlp
from pa_nlp.nlp import *

def run_port(options, args):
  if options.search is not None:
    cmd = f"port search {options.search}"

  elif options.install is not None:
    cmd = f"sudo port install {options.install}"

  elif options.uninstall is not None:
    cmd = f"sudo port uninstall {options.uninstall}"

  elif options.update:
    cmd = f"sudo port update"

  elif options.list_installed:
    cmd = f"port list"

  nlp.execute_cmd(cmd)

def run_apt(options, args):
  if options.search is not None:
    cmd = f"apt search {options.search}"

  elif options.install is not None:
    cmd = f"sudo apt install {options.install}"

  elif options.uninstall is not None:
    cmd = f"sudo apt remove {options.uninstall}"

  elif options.update:
    cmd = f"sudo apt update"

  elif options.list_installed:
    cmd = f"apt list --installed"

  nlp.execute_cmd(cmd)

def run_pip(options, args):
  py = f"python{options.python}"
  assert py is not None

  if options.search is not None:
    cmd = f"{py} -m pip search {options.search}"

  elif options.install is not None:
    cmd = f"pip{options.python} install {options.install} --user {options.user}"

  elif options.uninstall is not None:
    cmd = f"sudo pip{options.python} uninstall {options.uninstall}"

  elif options.upgrade:
    cmd = f"sudo {py} -m pip install pip --upgrade"

  elif options.list_installed:
    cmd = f"{py} -m pip list"

  nlp.execute_cmd(cmd)

def main():
  if platform.system() == "Linux":
    tool_default = "apt"
    tool_cands = "[apt*, pip]"
  else:
    tool_default = "port"
    tool_cands = "[port*, pip]"

  parser = OptionParser(usage = "cmd [optons] ")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose", \
      #default = False, help = "don't print status messages to stdout")
  parser.add_option("--tool", default=tool_default, help=tool_cands)
  parser.add_option("--search", help="")
  parser.add_option("--install", help="")
  parser.add_option("--uninstall", help="")
  parser.add_option("--update", action="store_true", help="")
  parser.add_option("--upgrade", action="store_true", help="")
  parser.add_option("--python", default="3", help="default 3")
  parser.add_option("--list_installed", action="store_true", help="")
  parser.add_option("--user", default="summer", help="")
  (options, args) = parser.parse_args()

  if options.tool == "port":
    run_port(options, args)

  elif options.tool == "apt":
    run_apt(options, args)

  elif options.tool == "pip":
    run_pip(options, args)

if __name__ == "__main__":
  main()

