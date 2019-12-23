#!/usr/bin/env python
#coding: utf8

from algorithm import *

if __name__ == "__main__":
  os.system("clear")

  parser = optparse.OptionParser(usage = "cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  (options, args) = parser.parse_args()

  # Removing all non-STL header files.
  for fname in open("header-list.txt"):
    fname = fname.strip()
    content = open(fname).read()\
        .replace("_GLIBCXX_VISIBILITY(default)", "")\
        .replace("_GLIBCXX_BEGIN_NAMESPACE_VERSION", "")\
        .replace("_GLIBCXX_END_NAMESPACE_VERSION", "")\
        .replace("_GLIBCXX_BEGIN_NAMESPACE_CONTAINER", "")\
        .replace("_GLIBCXX_END_NAMESPACE_CONTAINER", "")\
        .replace("_GLIBCXX_NOEXCEPT", "")\
        .replace("noexcept", "")
    print >> open(fname, "w"), content
    print "%s is finished" %fname





