#!/usr/bin/env python3

from pa_nlp import nlp
import optparse
import os
import time

if __name__ == "__main__":
  parser = optparse.OptionParser(usage = "cmd [optons]")
  (options, args) = parser.parse_args()

  cmd = "sudo lsof /dev/nvidia* | cut -f 2 -d ' '  | sort | uniq"
  threads = os.popen(cmd).read().split()
  for td in threads:
    cmd = f"sudo kill -9 {td}"
    os.system(cmd)

