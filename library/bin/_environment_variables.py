#!/usr/bin/env python3

from algorithm_3x import *
from pa_nlp import common as nlp
import os

if __name__ == "__main__":
  parser = OptionParser(usage = "cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose", default = False, help = "don't print status messages to stdout")
  (options, args) = parser.parse_args()

  keys = ["PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH",
          "PYTHONPATH"]
  for key in keys:
    values = os.environ.get(key, None)
    if values is None:
      print("No %s in the environment" %key)
      continue
    print("*" * 32, key, "*" * 32)
    values = [v.strip() for v in values.split(":")
              if not nlp.is_none_or_empty(v.strip())]
    for vi, v in enumerate(sorted(values)):
      print(vi, v)
    print()

