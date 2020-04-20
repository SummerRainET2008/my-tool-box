#!/usr/bin/env python3
#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

import tensorflow as tf

import numpy as np

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def save_to_file(samples: list, file_name: str):
  def serialize_example(file_name: str, label: int, mfcc: np.ndarray):
    feature = {
      'file': _bytes_feature(file_name.encode("utf8")),
      'label': _int64_feature(label),
      "mfcc": _bytes_feature(mfcc.tobytes()),
    }
    example_proto = tf.train.Example(
      features=tf.train.Features(feature=feature)
    )

    return example_proto.SerializeToString()

  with tf.python_io.TFRecordWriter(file_name) as writer:
    for sample in samples:
      example = serialize_example(*sample)
      writer.write(example)

def read_from_file(file_name: str):
  def parse_fn(example):
    example_fmt = {
      "file": tf.FixedLenFeature((), tf.string, ""),
      "label": tf.FixedLenFeature((), tf.int64, -1),
      "mfcc": tf.FixedLenFeature((), tf.string, "")
    }
    parsed = tf.parse_single_example(example, example_fmt)
    file_name = parsed["file"]
    mfcc = parsed["mfcc"]
    mfcc = tf.decode_raw(mfcc, tf.float32)

    return file_name, mfcc, parsed["label"]

  def input_fn():
    files = tf.data.Dataset.list_files("test.tfrecord")
    dataset = files.interleave(tf.data.TFRecordDataset, 2)
    dataset = dataset.shuffle(buffer_size=5)
    dataset = dataset.map(map_func=parse_fn)
    dataset = dataset.batch(batch_size=2)
    dataset = dataset.repeat(10)

    return dataset

  dataset = input_fn()
  data_iter = dataset.prefetch(10).make_initializable_iterator()
  sample = data_iter.get_next()

  return data_iter.initializer, sample

def main():
  samples = [
    ("file_1", 1, np.array([0.98761346923, 0.45], np.float32)),
    ("file_2", 2, np.array([324567.0000019876, 0.34], np.float32)),
    ("file_3", 3, np.array([324567.0000019876, 0.34], np.float32)),
  ]

  save_to_file(samples, "test.tfrecord")

  initializer, sample = read_from_file("test.tfrecord")

  sess = tf.Session()
  sess.run(initializer)

  while True:
    try:
      print(sess.run(sample))
    except tf.errors.OutOfRangeError:
      print("done")
      break

if __name__ == "__main__":
  main()