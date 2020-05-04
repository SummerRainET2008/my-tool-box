#!/usr/bin/env python3

from pa_nlp import nlp
from os import system
from optparse import OptionParser
from random import sample
from threading import Thread

class PlayThread:
  def __init__(self, play_cmd):
    self.cmd = play_cmd

  def __call__(self):
    system(self.cmd)

class Sound:
  home = nlp.get_home_dir()
  SOUND_DIR = f"{nlp.get_home_dir()}/my-tool-box/bin/music"
  SID = 0
  SOUND_TABLE = []

  def __init__(self, name):
    self.name = name
    self.sid = Sound.SID
    Sound.SID += 1

  @staticmethod
  def gen_all_sounds(level: str):
    types = [("-", "34567"), ("", "1234567"), ("+", "1234567"), ("++", "123")]

    ret = []
    for t, ns in types:
      for s in ns:
        name = "%s%s" % (s, t)
        ret.append(Sound(name))
        Sound.SOUND_TABLE.append(name)

        if s not in "37":
          ret.append(Sound("#" + name))
          Sound.SOUND_TABLE.append("#" + name)

    ret.append(Sound("4++"))
    Sound.SOUND_TABLE.append("4++")

    if level == "1":
      f, t = Sound.SOUND_TABLE.index("1"), Sound.SOUND_TABLE.index("7")
    elif level == "2":
      f, t = Sound.SOUND_TABLE.index("1"), Sound.SOUND_TABLE.index("7+")
    elif level == "3":
      f, t = Sound.SOUND_TABLE.index("-3"), Sound.SOUND_TABLE.index("7+")
    elif level == "4":
      f, t = 0, len(Sound.SOUND_TABLE) - 1

    return ret[f: t + 1]

  def play(self):
    # increase to standard.
    if self.name not in ["correct", "incorrect"]:
      idx = Sound.SOUND_TABLE.index(self.name) + 1
      name = Sound.SOUND_TABLE[idx]
    else:
      name = self.name
    cmd = "afplay -t 2 %s/%s.aiff" % (Sound.SOUND_DIR, name)
    Thread(target=PlayThread(cmd)).run()

class Music:
  def __init__(self, level: str):
    self._sounds = [sd for sd in Sound.gen_all_sounds(level)
                    if "#" not in sd.name]
    self._name2sounds = {
      sd.name: sd for sd in self._sounds
    }
    self._wrong_sound_hint = Sound("incorrect")
    self._right_sounds = Sound("correct")

  def learn(self):
    for sound in self._sounds * 100:
      key = input("push any key to halt, and push 'enter' to practice: ")
      if key.strip() != "":
        break
      print("name: ", sound.name)
      sound.play()

  def guess(self):
    while True:
      sound = sample(self._sounds, 1)[0]

      print([s.name for s in self._sounds])
      while True:
        print("listen...")
        sound.play()
        iname = input("input name: ").strip()
        if iname == sound.name:
          self._right_sounds.play()
          break
        else:
          guess_sound = self._name2sounds.get(iname)
          if guess_sound is not None:
            self._wrong_sound_hint.play()
            print(f"This is your guess")
            guess_sound.play()

  def compare(self):
    while True:
      while True:
        s1, s2 = sample(self._sounds, 2)
        if 0 < abs(s1.sid - s2.sid) <= 3:
          break
      print("listening...")

      while True:
        s1.play()
        s2.play()

        while True:
          v = input("input the maximum one: (1, 2): ").strip()
          if v in ["1", "2"]:
            break

        v = int(v)
        cmp_result = nlp.cmp(s1.sid, s2.sid)
        if (v == 1 and cmp_result == 1) or (v == 2 and cmp_result == -1):
          self._right_sounds.play()
          print(s1.name, s2.name)
          break
        else:
          self._wrong_sound_hint.play()

if __name__ == "__main__":
  parser = OptionParser(usage="cmd [optons] ..]")
  parser.add_option("--learn", action="store_true", 
                    help="learning the frequency of all sounds")
  parser.add_option("--guess", action="store_true", 
                    help="guess the sound when hearing")
  parser.add_option("--compare", action="store_true", help="compare")
  parser.add_option("--level", default="1", help="level: 1-4, default '1'")
  (options, args) = parser.parse_args()

  music = Music(options.level)
  if options.learn:
    music.learn()
  elif options.guess:
    music.guess()
  elif options.compare:
    music.compare()
