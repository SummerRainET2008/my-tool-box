#!/usr/bin/env python

import subprocess
import optparse   
import os
import popen2
import re
import string
import sys
import _thread
import time

debug = None

class VimPythonDebugger(object):
  def __init__(self, server_name, main_class):
    self._server_name   = server_name 
    self._quit          = False
    self._lock          = _thread.allocate_lock()
    self._main_class    = main_class 
    self._bp2file       = dict()
    self._file2bp       = dict()

  def system(self, cmd):
    self.log("cmd: %s" %cmd)
    os.system(cmd)

  def log(self, *msgs):
    if debug:
      print("\nlog:", end=' ') 
      for msg in msgs:
        print(msg, end=' ') 
      print()   

  def _parse_pdb_status(self, pdb_pipe, line):
    '''Parsing commands from pdb and send to Vim'''
    m1 = re.search('''Breakpoint (\d+) at (.*?):(\d+)''', line)
    m2 = re.search('''> ([^<>].*?)\((\d+)\)(<module>|[a-zA-Z_\d]+)\(\)''', line)
    m3 = re.search('''Deleted breakpoint (\d+)''', line)

    if m1 is not None:
      fname, lineID = m1.group(2), m1.group(3)
      self._file2bp[fname + ":" + lineID] = int(m1.group(1))
      self._bp2file[int(m1.group(1))] = fname + ":" + lineID

      self.log("Breakpoint Set Detected:", m1.groups())
      self._send_to_vim("VDBBreakSet(%s, '%s', %s)" %(lineID, fname, lineID))
      return

    if m2 is not None:
      fname, lineID = m2.group(1), m2.group(2)
      self.log("Line Step Detected:", m2.groups())
      self._send_to_vim("VDBHighlightLine(%s, '%s')" %(lineID, fname))
      return

    if m3 is not None:
      bp = int(m3.group(1))
      fname, lineID = self._bp2file[bp].split(":") 
      self.log("Breakpoint Clear Detected:", m3.groups())
      self._send_to_vim("VDBBreakClear(%s, '%s')" %(lineID, fname))
      return

    self.log("Ignoring '%s'" %line)

  def _monitor_vim(self, pdb_pipe):
    '''Monitoring commands from Vim from .{_server_name}, and writing to pdb'''
    while True:
      cmdPipe = os.popen("cat .%s" % self._server_name)
      for line in cmdPipe:
        self.log("_monitor_vim: '%s'" %line)
        
        if line.startswith("clear"):
          inf = " ".join(line.split()[1:])
          if inf not in self._file2bp:
            continue
          line = "clear %d\n" %self._file2bp[inf]

        pdb_pipe.tochild.write(line)
      cmdPipe.close()

      with self._lock:
        if self._quit:
          break

  def _monitor_pdb(self, pdb_pipe):
    '''Filtering status from pdb and send to Vim.'''
    line = ""
    while True:
      char = pdb_pipe.fromchild.read(1)
      sys.stdout.write(char)
      sys.stdout.flush()

      if char == '\n':
        self.log("_monitor_pdb: '%s'" %line)
        self._parse_pdb_status(pdb_pipe, line)
        line = ""
      else:
        line += char

      with self._lock:
        if self._quit: 
          break

  def _monitor_cmd_input(self, pdb_pipe):
    while True:
      char = sys.stdin.read(1)
      pdb_pipe.tochild.write(char)
      with self._lock:
        if self._quit:
          pdb_pipe.tochild.write("quit\n")
          break

  def _send_to_vim(self, cmd, silent = True):
    '''In cmd, only ' is permitted, instead of ".'''
    silent = "silent" if silent else ""
    self.system('''vim --servername %s -u NONE -U NONE '''\
                '''--remote-send "<C-\\><C-N>:%s call %s<CR><CR>"'''
                %(self._server_name, silent, cmd));

  def run(self, params):
    try:
      # Monitoring commands from Vim.
      os.mkfifo(".%s" % self._server_name, 0o600)
      self._send_to_vim('''VDBInit('.%s', '%s')''' 
                        %(self._server_name, os.path.realpath(os.curdir)))
      # Monitoring pdb.
      pdb = "python -m pdb %s '%s'" %(self._main_class, params)
      pdb_pipe = popen2.Popen3(pdb, capturestderr = False, bufsize = 0)

      _thread.start_new_thread(self._monitor_pdb, (pdb_pipe,))
      _thread.start_new_thread(self._monitor_cmd_input, (pdb_pipe,))
      _thread.start_new_thread(self._monitor_vim, (pdb_pipe,))

      try:
        while True:
          if pdb_pipe.poll() != -1:
            self._quit = True
            break
          time.sleep(0.1)
      except KeyboardInterrupt as ex:
        with self._lock:
          self._quit = True

      time.sleep(0.3)
      self._send_to_vim("VDBClose()")

    finally:
      self.system("rm -f .%s" % self._server_name)

if __name__ == "__main__":
  os.system("clear")

  parser = optparse.OptionParser(usage = "cmd [optons] mainclass")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  parser.add_option("--debug", action = "store_true", dest = "debug",
                     default = False, help = "debug mode")
  parser.add_option("--servername", dest = "server_name",
                     default = "debug", help = "default 'debug'")
  (options, args) = parser.parse_args()

  debug = options.debug
  main_class = args[0] if args != [] else "main1.py"
  VimPythonDebugger(options.server_name, main_class).run(" ".join(args))
