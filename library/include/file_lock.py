from palframe import common as nlp
import os
import sys
import time

class FileLock:
  def __init__(self, lock_name, sleepTime=60):
    '''sleepTime: seconds'''
    self._sleepTime = sleepTime
    self._lock_name = f"/tmp/lock.{lock_name}"
    self._waitUntilRelease()
    nlp.execute_cmd("touch %s" % lock_name)

  def unlock(self):
    nlp.execute_cmd("rm %s" % self._lock_name)

  def _waitUntilRelease(self):
    while os.path.isfile(self._lock_name):
      print("_waitUntilRelease ...")
      sys.stdout.flush()
      time.sleep(self._sleepTime)
