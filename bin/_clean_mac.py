#!/usr/bin/env python
#coding: utf8

from algorithm import *

garbages = [
  #"sudo port -f uninstall inactive",
  "sudo rm -r /opt/local/var/macports/distfiles/*",
  "sudo rm -r /opt/local/var/macports/build/*",
  "sudo rm -rf /opt/local/var/macports/packages/*",
  "sudo rm -rf /opt/local/var/macports/software/*",
]

# Take care of "/usr/local/texlive", whether it has history files to delete.

if __name__ == "__main__":
  os.system("clear")

  parser = optparse.OptionParser(usage = "cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  (options, args) = parser.parse_args()

  for cmd in garbages:
    os.system(cmd)


