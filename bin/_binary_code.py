#!/usr/bin/env python

from algorithm import *

def calc_binary_code(d):
  ret = []
  for i in xrange(32):
    ret.append('1' if d >> i & 1 == 1 else ' ')
  return list(reversed(ret))

if __name__ == "__main__":
  os.system("clear")

  usage = '''
  cmd -c "m = 10; n =20" "10 << 2; 20 + 30; -1; m + n"
  '''

  parser = optparse.OptionParser(usage = usage)
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose", \
      #default = False, help = "don't print status messages to stdout")
  parser.add_option("-c", "--condition", dest = "condition", \
      default = "", help = "")
  (options, args) = parser.parse_args()

  fmt = "%2s%2s%2s%2s%2s%2s%2s%2s%2s%2s%2s%2s%2s%2s%2s%2s%2s"\
        "%2s%2s%2s%2s%2s%2s%2s%2s%2s%2s%2s%2s%2s%2s%2s%12s%64s" 

  title = [e - (e / 10) * 10 for e in range(31, -1, -1)]
  print fmt %tuple(title + ["value", "clause"]), "\n"

  exec(options.condition)
  for clause in args[0].split(";"):
    exec("value = %s" %clause)
    print fmt %tuple(calc_binary_code(value) + [value] + [clause])


