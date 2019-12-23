#!/usr/bin/env python3

import tensorflow as tf

def test_1():
  print(f"\n\n{'-' * 36} test 1 {'-' * 36}")
  # Creates a graph.
  with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    e = tf.Variable([1.])

  c = tf.matmul(a, b)
  d = a + a

  config = tf.ConfigProto(
    log_device_placement=True, allow_soft_placement=True
  )
  config.gpu_options.allow_growth = True
  # config.gpu_options.per_process_gpu_memory_fraction = 0.4
  sess = tf.Session(config=config)

  print("a.type", a.op.type)
  print("b.type", b.op.type)
  print("c.type", c.op.type)
  print("d.type", d.op.type)
  print("e.type", e.op.type)
  print(sess.run(c))

def test_2():
  print(f"\n\n{'-' * 36} test 2 {'-' * 36}")
  config = tf.ConfigProto(
    log_device_placement=True, allow_soft_placement=True
  )
  config.gpu_options.allow_growth = True
  # config.gpu_options.per_process_gpu_memory_fraction = 0.4
  sess = tf.Session(config=config)

  x = tf.convert_to_tensor(list(range(16)), tf.float32)
  x = tf.reshape(x, [1, 16, 1])
  x = tf.layers.conv1d(
    x, 3, 2, kernel_initializer=tf.initializers.random_uniform(-1, 1)
  )
  sess.run(tf.global_variables_initializer())
  print(sess.run(x))

if __name__ == "__main__":
  print(f"tensorflow version: {tf.__version__}")

  test_1()
  test_2()



