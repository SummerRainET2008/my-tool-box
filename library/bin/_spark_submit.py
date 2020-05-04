#!/usr/bin/env python2
#coding: utf8

from algorithm import *

cmdTpt = ("/opt/spark/%s/bin/spark-submit "
          "--master yarn "
          "--conf spark.yarn.executor.memoryOverhead=%d "
          "--conf spark.driver.maxResultSize=%dG "
          "--conf spark.dynamicAllocation.maxExecutors=%d "
          "--driver-memory %dg "
          "--executor-memory %dg "
          "--py-files %s%s "
          "%s")

defaultIncludedFiles = [
  "/home/txia/.vim/include/algorithm.py",
  "/home/txia/.vim/include/PigDataSchema.py",
]

if __name__ == "__main__":
  os.system("clear")

  parser = optparse.OptionParser(usage = "cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  parser.add_option("--maxResultSize", type = int, default = 3,
                    dest = "maxResultSize", help = "3G")
  parser.add_option("--maxExecutors", type = int, default = 200,
                    dest = "maxExecutors", help = "200")
  parser.add_option("--driverMemory", type = int, default = 15,
                    dest = "driverMemory", help = "15G")
  parser.add_option("--executorMemory", type = int, default = 6,
                    dest = "executorMemory", help = "6g")
  parser.add_option("--memoryOverhead", type = int, default = 6,
                    dest = "memoryOverhead", help = "default 6G")
  parser.add_option("--pyFiles", default = "", dest = "pyFiles")
  parser.add_option("--version", default = "1.6.2", dest = "version",
                   help = "default 1.6.2")
  (options, args) = parser.parse_args()

  pyFiles = options.pyFiles.strip()
  pyFiles = "," + pyFiles if pyFiles != "" else ""

  cmd = cmdTpt %(options.version,
                 options.memoryOverhead * 1024,
                 options.maxResultSize, options.maxExecutors,
                 options.driverMemory, options.executorMemory,
                 ",".join(defaultIncludedFiles), pyFiles,
                 " ".join(args))

  print(cmd)
  executeCmd(cmd)

