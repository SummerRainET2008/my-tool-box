#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

from palframe import common as nlp
import time
import multiprocessing as mp
from palframe.audio.audio_helper import AudioHelper
import optparse
import collections
import os

def check_1(folder: str):
  mp3_files = [os.path.basename(fn).rpartition(".")[0]
               for fn in nlp.get_files_in_folder(folder, ["mp3"])]
  txt_files = [os.path.basename(fn).rpartition(".")[0]
               for fn in nlp.get_files_in_folder(folder, ["txt"])]

  mp3_counts = collections.Counter(mp3_files)
  txt_counts = collections.Counter(txt_files)
  if (len(mp3_files) == len(txt_files) and
    len(mp3_counts) == len(txt_counts)):
    return

  for mp3 in mp3_counts:
    if mp3 not in txt_files:
      print(f"ERR: no {mp3}.txt found")

  for txt in txt_files:
    if txt not in mp3_counts:
      print(f"ERR: no {mp3}.mp3 found")

def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--data_folder")
  #default=False, help="")
  (options, args) = parser.parse_args()

  check_1(options.data_folder)

if __name__ == '__main__':
  main()

