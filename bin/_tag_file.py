#! /usr/bin/env python2.7

from algorithm import *
import subprocess

debug = False
#debug = True

class FileTag:
  def __init__(self, fn):
    self._fn = fn

  def read_tags(self):
    return []

  def append_tags(self, tags):
    # tags: a list
    tags = self.read_tags() + tags
    self._write_tags(tags)

  def _write_tags(self, tags):
    # writes the list of tags to three xattr fields on a file-by file basis:
    # [kMDItemFinderComment, _kMDItemUserTags, kMDItemOMUserTags].
    # Uses subprocess instead of xattr module. Slower but no dependencies.
    # To check if it worked, use xattr -lx FileName 

    plist_tpt = '''<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" \
      "http://www.apple.com/DTDs/PropertyList-1.0.dtd"> \
      <plist version="1.0"> <array> %s </array> </plist> '''

    plist_tags = " ".join(["<string> %s </string>" %tag for tag in tags])
    plist_text = (plist_tpt %plist_tags).encode("utf8")

    opt_tag = "com.apple.metadata:"
    result = ""
    for field in [
                  #"kMDItemFinderComment", 
                  "_kMDItemUserTags", 
                  #"kMDItemOMUserTags"
                 ]:
      xattr_cmd = '''xattr -w %s '%s' "%s"''' %(opt_tag + field, 
                                                plist_text, 
                                                self._fn)
      if debug:
        print xattr_cmd, "\n"
      r = subprocess.check_output(xattr_cmd, shell = True, 
                                  stderr = subprocess.STDOUT)
      result += r
    return result

if __name__ == "__main__":
  parser = optparse.OptionParser(usage = "cmd [optons] 'tag1 tag2 ...'" 
                                 "file1 file2 ...")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  (options, args) = parser.parse_args()
  if len(args) >= 2:
    tags = args[0].split() 
    for f in args[1:]: 
      file_tag = FileTag(f)
      file_tag.append_tags(tags)
      print "%s is done" %f     
