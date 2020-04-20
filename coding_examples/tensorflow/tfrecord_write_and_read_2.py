#!/usr/bin/env python3
#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

import time
import tensorflow as tf
import pa_nlp.tensorflow as nlp_tf

import numpy as np

def write_data():
  samples = [
    ("file_1", 1, np.array([0.98761346923, 0.45]     * 100_000, np.float32)),
    ("file_2", 2, np.array([324567.0000019876, 0.34] * 100_000, np.float32)),
    ("file_3", 3, np.array([324567.0000019876, 0.34] * 100_000, np.float32)),
  ]
  data = samples * 1000

  def serialize_sample_fun(sample):
    file_name, label, mfcc = sample
    feature = {
      'file':   nlp_tf._bytes_feature(file_name.encode("utf8")),
      'label':  nlp_tf. _int64_feature(label),
      "mfcc":   nlp_tf._bytes_feature(mfcc.tostring()),
    }
    example_proto = tf.train.Example(
      features=tf.train.Features(feature=feature)
    )

    return example_proto.SerializeToString()

  start_time = time.time()
  nlp_tf.save_to_file(iter(data), serialize_sample_fun, "test.tfrecord")
  duration = time.time() - start_time
  print(f"writing time: {duration} seconds.")

def read_data():
  example_fmt = {
    "file": tf.FixedLenFeature((), tf.string, ""),
    "label": tf.FixedLenFeature((), tf.int64, -1),
    "mfcc": tf.FixedLenFeature((), tf.string, "")
  }

  def example2sample_func(parsed):
    file_name = parsed["file"]
    mfcc = parsed["mfcc"]
    mfcc = tf.decode_raw(mfcc, tf.float32)

    return file_name, mfcc, parsed["label"]

  initializer, batch_tf = nlp_tf.read_tfrecord_file(
    "test.tfrecord", example_fmt, example2sample_func,
    10, 48,
  )

  sess = tf.Session()
  sess.run(initializer)

  num = 0
  while True:
    try:
      start_time = time.time()
      batch = sess.run(batch_tf)
      duration = time.time() - start_time
      print(f"fetch time: {duration} seconds.")
      num += 1

      if num == 10:
        break
    except tf.errors.OutOfRangeError:
      print("done")
      break

def main():
  write_data()
  read_data()

if __name__ == "__main__":
  main()