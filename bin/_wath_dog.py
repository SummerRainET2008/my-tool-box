#!/usr/bin/env python
#coding: utf8

import os
import optparse
import time

def is_running(thread_name):
  running_threads = "\n".join(os.popen("ps -A"))
  return thread_name in running_threads

def ensure_run_command(cmd, thread_name):
  while not is_running(thread_name):
    print "staring ... ", options.cmd
    os.system(cmd)
    time.sleep(30)

if __name__ == "__main__":
  os.system("clear")

  parser = optparse.OptionParser(usage = "cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  parser.add_option("--thread_name", dest = "thread_name",
                    help = "thread name to watch")
  parser.add_option("--cmd", dest = "cmd")
  parser.add_option("--est_time", dest = "est_time", type = float,
                    help = "estimated hours to run the target thread")
  (options, args) = parser.parse_args()

  if "nohup" not in options.cmd:
    options.cmd = "nohup %s &" %options.cmd

  est_minutes = options.est_time * 60
  sleep_time = 5  # minutes

  while est_minutes > 0:
    ensure_run_command(options.cmd, options.thread_name)
    print "sleeping %d minutes ..." %sleep_time
    time.sleep(60 * sleep_time)
    est_minutes -= sleep_time
