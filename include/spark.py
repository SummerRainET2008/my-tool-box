#!/usr/bin/env python

from pa_nlp import common as nlp
import operator

class Spark:
  @staticmethod
  def hadoop_delete_file(fname):
    nlp.execute_cmd("hadoop fs -rm -r %s" %fname)

  @staticmethod
  def to_utf8(data, keys: list):
    def mapper(vd):
      for key in keys:
        value = nlp.to_utf8(vd[key])
        if value is None:
          return None
        vd[key] = value
      return vd    

    return data.map(mapper).filter(lambda vd: vd is not None)

  @staticmethod
  def sql(hiveContext, sql):
    '''from pyspark import SparkContext, HiveContext
       sc = SparkContext()
       hiveContext = HiveContext(sc)
       Please remember string value from database might be unicode
       '''
    return hiveContext.sql(sql).rdd.coalesce(1024, True)

  @staticmethod
  def save(data, outputDir):
    Spark.hadoopDeleteFile(outputDir)
    data.saveAsTextFile(outputDir)

  @staticmethod
  def read_pig_data(sc, fname, schema, blockSize = 1024):
    '''We should add .coalesce(1024, True) for any read operation.
    '''
    def mapper(line):
      line = nlp.to_utf8(line)
      if line is None:
        return None

      values = line.split("\t")
      if len(values) == 0:
        return None
      return dict(list(zip(schema, values)))

    return sc.textFile(fname).coalesce(blockSize, True)\
             .map(mapper).filter(lambda vd: vd is not None)

  @staticmethod
  def read_object_data(sc, fname, blockSize = 1024):
    def evalObject(ln):
      try:
        return eval(ln)
      except Exception as error:
        print(error)
        print("ln:", ln) 
        assert False

    return sc.textFile(fname).coalesce(blockSize, True)\
             .map(evalObject)

  @staticmethod
  def map_to_key_value(data, keys):
    return data.map(lambda vd: (Spark.get_key(vd, keys), vd))

  @staticmethod
  def map_to_key_value_list(data, keys):
    return data.map(lambda vd: (Spark.get_key(vd, keys), [vd]))

  @staticmethod
  def group_by_key(data, keys):
    return data.map(lambda vd: (Spark.get_key(vd, keys), [vd]))\
               .reduceByKey(operator.add)

  @staticmethod
  def get_key(vd, keys):
    '''Only utf8 string, or int'''
    return "+".join(str(vd.get(key, "")) for key in keys)

  @staticmethod
  def distinct_by_key(data, keys):
    return list(Spark.map_to_key_value(data, keys)\
                .reduceByKey(lambda vd1, vd2: vd1).values())

  @staticmethod
  def removeNullKeyValue(data, keys):
    '''If some key-value is null or empty, then the final key-string
    would make no sense.'''
    return data.filter(lambda vd: not any([nlp.is_none_or_empty(vd.get(key))
                                           for key in keys]))

  @staticmethod
  def intersec_by_key(data1, data2, keys):
    '''We must guarantee data == Spark.distinctByKey(data, keys)
    '''
    data1 = Spark.map_to_key_value(data1, keys)
    data2 = Spark.map_to_key_value(data2, keys)
    return list(data1.join(data2).values())

  @staticmethod
  def union_by_key(sc, datas, keys):
    return Spark.distinct_by_key(sc.union(datas), keys)

