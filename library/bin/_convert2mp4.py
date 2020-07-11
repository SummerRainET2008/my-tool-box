#!/usr/bin/env python3

from pa_nlp import nlp
from pa_nlp.nlp import Logger
from pa_nlp import *

def main():
  parser = OptionParser(usage = "cmd [optons] file1 file2 ...]")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  (options, args) = parser.parse_args()

  for f in args:
    ext = nlp.get_file_extension(f)
    if ext.lower() == "mp4":
      Logger.warn(f"The input file '{f}' is already mp4 file.")
      continue

    new_f = nlp.replace_file_name(f, ext, "mp4")
    nlp.execute_cmd(f"ffmpeg -i {f} -q:a 0 -r 30 -strict -2 {new_f}")

if __name__ == "__main__":
  main()
