#!/usr/bin/env python

from algorithm import *

reg = re.compile("(\d+)\.(\d+)")

def normalize_figure_name(old_name):
  path, _, old_short_name = old_name.rpartition("/")
  old_short_name, _, extension = old_short_name.rpartition(".")
  
  new_short_name = reg\
      .sub(r"\1_dot_\2", old_short_name)\
      .replace(".", "___")\

  if path != "":
    return path + "/" + new_short_name + "." + extension
  else:
    return new_short_name + "." + extension

def get_new_lyx_filename(fname):
  assert fname.endswith(".lyx")
  fname, _, _ = fname.rpartition(".")
  return fname + ".convered.lyx"

def update_lyx_file(fn_old, fn_new):
  reg_tag = re.compile("^(\s*filename\s+)(.*)")
  fi, fo = open(fn_old), open(fn_new, "w")
  tag = "filename"
  for ln in fi:
    result = reg_tag.findall(ln)
    if result != []:
      filename = result[0][1] 
      new_filename = normalize_figure_name(filename)
      print >> fo, "%s%s" %(result[0][0], new_filename)
    else:
      print >> fo, ln,
  fo.close()
  print "%s is done" %(fn_new)

if __name__ == "__main__":
  parser = OptionParser(usage = "cmd [optons] file1.lyx file2.lyx ...]")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose", \
      #default = False, help = "don't print status messages to stdout")
  parser.add_option("-t", "--test", dest = "fn_test", \
      default = None, help = '''--test "filename"''')
  (options, args) = parser.parse_args()

  filenames = [
    "./baselines.4.microsoft.10K/curve/lambda-mart.log.leaf_10.l_0.1.ERR.fold1.pdf",
    "./baselines.4.microsoft.10K/improvements-curve/improvements.lambda-mart.log.leaf_10.l_0.12.pdf.pdf",
  ]

  print "Show some examples first"
  for fname in filenames:
    print "%s\n%s\n" %(fname, normalize_figure_name(fname))
  print "\n\n"

  if options.fn_test is not None:
    print "%s\n%s\n" %(options.fn_test, normalize_figure_name(options.fn_test))
  print "\n\n"

  for fname in args:
    update_lyx_file(fname, get_new_lyx_filename(fname))

