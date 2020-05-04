#!/usr/bin/env python
#coding: utf8

'''Rename paper names following such rules.
[year].[type = paper, slide, note, long].[conference].[key words].filename.pdf

I  put  the  year  first  based  on  the  reason  that  I  should  focuse on the
state-of-the-art techqniues.
'''

from algorithm import *

conferences = [
  "aaai", "acl", "aistats", "aistatistics", "amta", "cikm", "cl", 
  "coling", "colling", "colt", "eacl", "eccv", "emnlp", "icassp",
  "iccl", "icml", "nips", "ieee", "ijcai", "ir", "jmlr", "kdd",
  "long", "mit", "ml", "naacl", "sigir", "uai", "wmt", "workshop",
  "www", "cvpr", "bionlp", "icdm", 
]

def analyze_tag(tag):
  tag = " ".join(tag.lower().replace(",", " ").split())
  info = {
    "year": None,
    "conference": None,
    "type": None, #slide, book
  }

  unknown = False
  for tok in tag.split():
    if tok in ["slide", "book", "dissertation", "note"]:
      info["type"] = tok.lower()
    elif len(tok) == 4 and tok.isdigit():
      info["year"] = tok
    elif tok in conferences:
      info["conference"] = tok
    else:
      #print "Ignoring", tok
      unknown = True
  if unknown:
    return None
  return info

def process_file(fname):
  path, fn = os.path.split(fname)
  if not fn.startswith("["):
    #print "Not '[]', so pass", fname
    return

  p = fn.find("]")
  assert p != -1
  tag, fn = fn[1: p], fn[p + 1:]
  info = analyze_tag(tag)
  if info is not None:
    #print path, info, fn
    new_fname = "%s/{year}{type}{conference}%s" %(path, fn)
    if info["year"] is None:
      new_fname = new_fname.replace("{year}", "")
    else: 
      new_fname = new_fname.replace("{year}", info["year"] + ".")
    if info["type"] is None:   
      new_fname = new_fname.replace("{type}", "")
    else: 
      new_fname = new_fname.replace("{type}", info["type"] + ".")
    if info["conference"] is None:   
      new_fname = new_fname.replace("{conference}", "")
    else: 
      new_fname = new_fname.replace("{conference}", info["conference"] + ".")
    print fname, "-->", new_fname     
    os.rename(fname, new_fname)
  else:
    print "Unknown:", fname

if __name__ == "__main__":
  os.system("clear")

  parser = optparse.OptionParser(usage = "cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  (options, args) = parser.parse_args()

  statis = collections.defaultdict(list)
  for fname in os.popen("find . -iregex '.*pdf'"):
    process_file(fname.strip())

