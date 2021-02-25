#!/usr/bin/env python3
#coding: utf8

from palframe import nlp
from palframe import *

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

def include_all_keywords(f: str, keywords: list):
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
  parser.add_option("--sort_by_size", action="store_true", default=False)
  parser.add_option("--min_file_size", type=int, default=0,
                    help="default 0 Mb.")
  parser.add_option("--cmd", help='e.g., chmod a-w {}')
  (options, args) = parser.parse_args()

  args = [kw.lower() for kw in args]
  if options.file_types == "":
    file_types = set([])
  else:
    file_types = set(options.file_types.split(","))

  valid_files = []
  for f in get_files_or_folders(options.dir_only,
                                options.file_only,
                                file_types,
                                options.global_search):
    if ".git" in f or "/." in f:
      continue

    if not options.reverse and not include_all_keywords(f, args) or \
      options.reverse and include_all_keywords(f, args):
      continue

    file_size = os.path.getsize(f) / 1024 / 1024
    if file_size < options.min_file_size:
      continue

    cmd = options.cmd
    if not nlp.is_none_or_empty(cmd):
      cmd = cmd.replace("{}", f"'{f}'")
    else:
      cmd = None

    result = {
      "file": f,
      "file_size": file_size,
      "cmd": cmd
    }
    valid_files.append(result)

  valid_files.sort(key=lambda item: -item["file_size"])
  for file_id, item in enumerate(valid_files):
    print(f"{file_id:10}: {item['file_size']:.2f} Mb, {item['file']}")
    cmd = item["cmd"]
    if cmd is not None:
      nlp.execute_cmd(item["cmd"])

if __name__ == "__main__":
  main()
