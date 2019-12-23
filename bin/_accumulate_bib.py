#!/usr/bin/env python

'''1. All reference files must be "references.bib".
'''

import sys
from algorithm import *

class Item(object):
  def __init__(self):
      self.type = None
      self.key  = None
      self.doc  = None

  @staticmethod
  def analyze_item(doc):
      markers = Item._analyze_parentheses(doc)
      last_pos = 0
      for begin, end in markers:
          p = doc.rfind("@", last_pos, begin)
          item = Item()
          item.type = doc[p: begin]
          item.key  = doc[begin + 1: doc.find(",", begin)]
          item.doc  = doc[p: end + 1]
          last_pos  = end
          yield item

  @staticmethod
  def _analyze_parentheses(doc):
      ret = []
      matched = 0
      for pos, ch in enumerate(doc):
          if ch == "{":
              if matched == 0:
                  ret.append([pos])
              matched += 1
          elif ch == "}":
              matched -= 1
              if matched == 0:
                  ret[-1].append(pos)
      return ret                
    
if __name__ == "__main__":
  parser = OptionParser(usage = "cmd [optons] ")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose", \
      #default = False, help = "don't print status messages to stdout")
  (options, args) = parser.parse_args()

  all_citations = {}
  for fname in popen('''find . -name "references.bib"'''):
    fname = fname.strip()
    print fname
    items = list(Item.analyze_item(open(fname).read()))
    for item in items:
      all_citations[item.key] = item
  print "There are %d citations" %(len(all_citations)) 

  all_citations = all_citations.values()
  all_citations.sort(key = lambda h: h.key)

  fou = open("references.all.bib", "w")
  for item in all_citations:
    print >> fou, item.doc
    print >> fou
  fou.close()     
    
