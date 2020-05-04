#!/usr/bin/env python3
#coding: utf8

from pa_nlp import nlp
import os
import optparse
import multiprocessing

VALID_FILE_EXTS = set([
  "pdf",
  "html", "webarchive",
  "jpg", "jpeg", "png",
  "numbers", "pages",
  "text",
  "mp4",
])

def list_files():
  def isCandidate(name):
    if "/." in name:
      return False
    if "Photos Library" in name:
      return False

    return name.rpartition(".")[2] in VALID_FILE_EXTS

  allFiles = os.popen("find .").read().split("\n")
  return list(filter(isCandidate, allFiles))

def normalize_file(fn):
  return fn.strip().replace('"', r'\"')

def analyze_file_tag(fname):
  fname = normalize_file(fname)
  try:
    line = os.popen('''_tag_file -l "%s"''' % fname).readlines()[0]
  except:
    print("skipping '%s'" % fname)
    return None

  toks = line.split("\t")
  if len(toks) == 1:
    #print(f"not found for '{fname}'")
    return None

  tags = toks[1].strip()
  print("found: '%s': %s" %(fname, tags))

  return fname, tags

def apply_file_tag(item):
  fn, tag = item
  assert len(file2tags) > 0
  fn = normalize_file(fn)
  try:
    print("adding '%s' into '%s'" %(tag, fn))
    cmd = '''_tag_file -s "%s" "%s"''' %(tag, fn)
    nlp.execute_cmd(cmd)
  except:
    print("failed '%s'" %fn)
  print()

if __name__ == "__main__":
  parser = optparse.OptionParser(usage = "cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  parser.add_option("--cmd", default="analyze",
                    help="[gen, apply, analyze], default 'analyze'")

  (options, args) = parser.parse_args()

  assert options.cmd in ["gen", "apply", "analyze"]

  tag_file = os.path.expanduser("~/my-tool-box/settings/tags.dict")
  os.chdir(os.path.expanduser("~/inf"))

  if options.cmd == "analyze":
    file2tags = eval(open(tag_file).read())
    folders = sorted(set([os.path.split(fn)[0] for fn in file2tags]))
    print("folders with tagged files")
    print("\n".join(folders))

  elif options.cmd == "gen":
    candFiles = list_files()
    print("#all files:", len(candFiles))
    print("permitted file extension:", VALID_FILE_EXTS)

    pmap = multiprocessing.Pool().map
    file2tags = dict([r for r in pmap(analyze_file_tag, candFiles)
                      if r is not None])
    print("#tagged files:", len(file2tags))
    print(file2tags, file=open(tag_file, "w"))

  elif options.cmd == "apply":
    file2tags = eval(open(tag_file).read())
    list(map(apply_file_tag, list(file2tags.items())))
    
