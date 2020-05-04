#!/usr/bin/env python
#coding: utf8

from algorithm import *
import codecs
import sys

class RenameTableTennies:
  videoExts = [".mp4"]

  def __init__(self):
    self._players = [
      ["马龙", "Ma Long"],
      ["马琳", "Ma Lin"],
      ["张继科", "Zhang Jike"],
      ["张怡宁", "Zhang Yining"],
      ["王皓", "Wang Hao"],
      ["王励勤", "Wang Liqin"],
      ["孔令辉", "Kong Linghui"],
      ["许昕", "Xu Xin"],
      ["方博", "Fang Bo"],
      ["樊振东", "Fan Zhendong"],
      ["庄智渊", "Chuang Chih Yuan"],
      ["陈玘", "Chen Qi"],
      ["波尔", "Timo Boll"],
      ["周雨", "Zhou Yu"],
      ["刘国梁", "Liu Guoliang"],
      ["奥恰洛夫", "Dimitrij Ovtcharov"],
      ["奥恰洛夫", "Ovtcharov"],
      ["平野美宇", "miu hirano"],
      ["老瓦", "waldner"],
    ]

    self._others = [
      ["-", " vs "],
      ["_", " "],
      ["[", "("],
      ["]", ")"],
      ["【", "("],
      ["】", ")"],
      ["（", "("],
      ["）", ")"],
      ["!", ""],
      ["！", ""],
      ["：", ": "],
    ]

  def _getVideos(self, files):
    for fname in files:
      lfname = fname.lower()
      for ext in RenameTableTennies.videoExts:
        if lfname.endswith(ext):
          yield fname

  def rename(self, files, debug):
    existent_vfiles = set(self._getVideos(os.listdir(".")))
    fou = open("log.rename.txt", "w")
    fid = 0

    for fn in self._getVideos(files):
      fn_new = self._getNewFile(fn)
      if fn_new == fn:
        continue

      msg = "%3d) %s --> %s" %(fid, fn, fn_new)
      print(msg) 
      print(msg, file=fou) 

      if fn_new in existent_vfiles:
        print("Error: new file name exists", file=fou)
        print("Error: new file name exists")
        break
      if not debug:
        os.rename(fn, fn_new)
        existent_vfiles.add(fn_new)
      fid += 1   
    fou.close()

  def _getNewFile(self, fn):
    for en_name, ch_name in self._players:
      fn = re.sub(".*".join(en_name.split()), ch_name, fn, 
                  flags = re.IGNORECASE)

    for str1, str2 in self._others:
      fn = fn.replace(str1, str2)

    fn = " ".join(fn.lower().split())
    return fn

if __name__ == "__main__":
  os.system("clear")

  parser = optparse.OptionParser(usage = "cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                    #default = False, help = "")
  parser.add_option("-D", action = "store_false", dest = "debug",
                    default = True, help = "No debug")
  (options, args) = parser.parse_args()

  os.chdir(os.getcwd())
  files = os.listdir(".")

  renamer = RenameTableTennies()
  renamer.rename(files, options.debug)
