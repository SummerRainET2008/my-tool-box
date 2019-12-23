#!/usr/bin/env python
#coding: utf8

import commands
import optparse   
import os
import popen2
import re
import string
import sys
import thread
import time
from xml.etree import ElementTree 

# Do NOT change here. Set in the cmd.
debug = None

class JDB(object):
  def __init__(self, src):
    self._lock = thread.allocate_lock()
    self._quit = False
    self._src  = src

  def _system(self, cmd):
    if debug:
      print "cmd: %s" %cmd
    os.system(cmd)

  def _monitor_jdb(self, jdb_pipe):
    '''Filtering status from jdb and print to console.'''
    while True:
      char = jdb_pipe.fromchild.read(1)
      sys.stdout.write(char)
      sys.stdout.flush()

      with self._lock:
        if self._quit: 
          break

  def _monitor_cmd_input(self, jdb_pipe):
    cur_cmd = ""
    while True:
      char = sys.stdin.read(1)
      if char == '\n':
        if cur_cmd != "":
          jdb_pipe.tochild.write('\n')
          cur_cmd = ""
        else:
          jdb_pipe.tochild.write("!!\n")
      else:
        jdb_pipe.tochild.write(char)
        cur_cmd += char
           
      with self._lock:
        if self._quit:
          jdb_pipe.tochild.write("quit\n")
          break

  def _set_classpath(self):
    if not os.path.exists(".classpath"):
      return

    lib_paths = []
    for node in ElementTree.parse(".classpath").findall("classpathentry"):
      if node.get("kind") in ["lib", "var", "output"]:
        lib_paths.append(node.get("path"))
      elif node.get("kind") == "src" and "output" in node.keys():
        lib_paths.append(node.get("output"))

    lib_paths = ":".join(lib_paths).replace("M2_REPO", 
                                            "/Users/%s/.m2/repository" 
                                            %os.getlogin())
    os.environ["CLASSPATH"] += ":" + lib_paths
    if debug:
      print os.environ["CLASSPATH"]

  def run(self, args):
    try:
      self._set_classpath()

      # Monitoring jdb.
      jdb = ("java jline.ConsoleRunner com.sun.tools.example.debug.tty.TTY "
             "-sourcepath .:%s %s" %(self._src, " ".join(args)))
      jdb_pipe = popen2.Popen3(jdb, capturestderr = False, bufsize = 0)

      thread.start_new_thread(self._monitor_jdb, (jdb_pipe,))
      thread.start_new_thread(self._monitor_cmd_input, (jdb_pipe,))

      try:
        while True:
          if jdb_pipe.poll() != -1:
            self._quit = True
            break
          time.sleep(1)
      finally:
        with self._lock:
          self._quit = True

    finally:
      print "exit"

if __name__ == "__main__":
  os.system("clear")
  parser = optparse.OptionParser(usage = "cmd [optons] main-class [parameters]")
  parser.add_option("--sourcepath", dest = "src", default = "")
  (options, args) = parser.parse_args()

  JDB(options.src).run(args)
