#!/usr/bin/env python3
#coding: utf8

from pa_nlp import nlp
from pa_nlp import *

def get_files_or_folders(dir_only: bool, file_only: bool, file_types: set,
                         global_search: bool):
  def exec_cmd(cmd):
    for f in os.popen(cmd):
      yield f.strip()

  if global_search:
    os.chdir("/Users/%s/inf" %os.getlogin())

  if dir_only:
    assert not file_only
    assert nlp.is_none_or_empty(file_types)

    cmd = f"find . -type d"
    yield from exec_cmd(cmd)

  else:
    if file_only:
      cmd = f"find . -type f"
    else:
      cmd = f"find ."

    for f in exec_cmd(cmd):
      # print("debug2:", f, nlp.is_none_or_empty(file_types), file_types)
      if nlp.is_none_or_empty(file_types) or \
        nlp.get_file_extension(f) in file_types:
        yield f

def include_keywords(f: str, keywords: list):
  f = f.split("/")[-1].lower()
  for kw in keywords:
    if kw not in f:
      return False

  return True

def main():
  parser = OptionParser(usage = "cmd [optons] kw1 kw2 ...]")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  parser.add_option("--dir_only", action="store_true", default=False)
  parser.add_option("--file_only", action="store_true", default=False)
  parser.add_option("--file_types", default="", help="such as py,cpp")
  parser.add_option("--global_search", action="store_true", default=False)
  parser.add_option("--reverse", action="store_true", default=False)
  parser.add_option("--cmd", help='e.g., chmod a-w {}')
  (options, args) = parser.parse_args()

  args = [kw.lower() for kw in args]
  if options.file_types == "":
    file_types = set([])
  else:
    file_types = set(options.file_types.split(","))

  num = 0
  for f in get_files_or_folders(options.dir_only,
                                options.file_only,
                                file_types,
                                options.global_search):
    # print("debug:", f)
    if ".git" in f or "/." in f:
      continue

    valid = include_keywords(f, args)
    if options.reverse:
      valid = not valid

    if valid:
      print(f"{num:8}: '{f}'")
      num += 1

      cmd = options.cmd
      if nlp.is_none_or_empty(cmd):
        continue
      cmd = cmd.replace("{}", f"'{f}'")
      # print(cmd)
      os.system(cmd)

if __name__ == "__main__":
  main()
