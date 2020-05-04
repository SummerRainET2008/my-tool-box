#!/usr/bin/env python3
#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

import time
import optparse

def run(size):
  start_time = time.time()
  result = 0
  for a in range(size):
    for b in range(size):
      for c in range(size):
        result += a + b + c

  end_time = time.time()
  print(f"It takes {end_time - start_time: .4f} s")
  print(f"GPU2, GPU3 machines: 1.799 s")
  print(f"MacBook Pro        : 2.224 s")

if __name__ == "__main__":
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
                     #default=False, help="")
  parser.add_option("--size", default=300)
  (options, args) = parser.parse_args()

  run(options.size)

