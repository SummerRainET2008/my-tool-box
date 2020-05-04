#!/usr/bin/env python

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
debug = False
#debug = True

class JavaRunner(object):
  def __init__(self):
    self._main_class = self._get_main_class()

  def _get_main_class(self):
    if os.path.isfile(".settings/org.eclim.prefs"):
      txt = open(".settings/org.eclim.prefs").read()
      rst = re.findall("org.eclim.java.run.mainclass=([^\s]+)", txt)
      if rst != []:
        return rst[0] 
    return "Main"    

  def system(self, cmd):
    if debug:
      print "cmd: %s" %cmd
    os.system(cmd)

  def _set_classpath(self):
    if not os.path.exists(".classpath"):
      return

    lib_paths = []
    for node in ElementTree.parse(".classpath").findall("classpathentry"):
      if node.get("kind") in ["lib", "var", "output"]:
        lib_paths.append(node.get("path"))
      elif node.get("kind") == "src":
        if "output" in node.keys():
          lib_paths.append(node.get("output"))

    lib_paths = ":".join(lib_paths).replace("M2_REPO", 
                                            "/Users/%s/.m2/repository" 
                                            %os.getlogin())
    os.environ["CLASSPATH"] += ":" + lib_paths
    if debug:
      print os.environ["CLASSPATH"]

  def run(self, params):
      self._set_classpath()
      cmd = "java -ea %s %s" %(self._main_class, params)
      self.system(cmd)

if __name__ == "__main__":
  os.system("clear")
  JavaRunner().run(" ".join(sys.argv[1:]))
