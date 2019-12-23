#!/usr/bin/env python

from algorithm import *

pyScript = '''
#!/usr/bin/python27-virtual-hadoop
#coding: utf8

from pyspark import SparkContext, HiveContext

if __name__ == "__main__":
  sc = SparkContext()
  hiveContext = HiveContext(sc)

  samples = hiveContext.sql("from %s select *").rdd.take(10)
  for si, sample in enumerate(samples):
    for key, value in sample.asDict().items():
      print "#sample:", si, "key=", key, "value=", value
    print  
  sc.stop()
'''

if __name__ == "__main__":
  parser = optparse.OptionParser(usage = "cmd [optons] test1 test2 ...]")
  parser.add_option("--table", dest = "table", default = None,
                   help = "database.table")
  (options, args) = parser.parse_args()

  table = options.table
  assert table is not None
  print >> open("/tmp/desc.hive.table.py", "w"), pyScript %table

  executeCmd("nohup _spark_submit.py --maxExecutors 100 "
             "--version 1.6.2 /tmp/desc.hive.table.py > desc.table.%s &" %table)
