#!/usr/bin/env python
#coding: utf-8

'''
sudo apt-get install python-mutagen
'''

from algorithm import *
import re
from sys import argv
from mutagen.mp3 import MP3 
from mutagen.easyid3 import EasyID3 
import mutagen.id3

def thread_update(param):
  fn, album = param
  album = fn if album is None else album
  m     = MP3(fn, ID3 = EasyID3) 
  try: 
    m.add_tags(ID3 = EasyID3) 
  except mutagen.id3.error as message:
    print message
  
  m["title"]  = fn.decode("utf8")
  m["album"]  = album.decode("utf8")
  m["artist"] = "" 
  m["genre"]  = ""
  m.save()
  print "%s is done" %fn

if __name__ == "__main__":
  usage = '''
  _my_update_mp3_ID.py `ls *.mp3` -a "王立群读史记--秦始皇"
  '''

  parser = OptionParser(usage = usage.decode("utf8"))
  #parser.add_option("-q", "--quiet", action = "store_true", \
      #dest = "verbose", default = False, \
      #help = "don't print status messages to stdout")
  parser.add_option("-a", "--album", dest = "album", \
      default = None, \
      help = "don't print status messages to stdout")
  (options, args) = parser.parse_args()
 
  p = Pool()
  p.map(thread_update, zip(args, [options.album] * len(args)))
  p.close()                 # add this 
  p.join()                  # add this
