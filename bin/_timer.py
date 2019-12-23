#!/usr/bin/env python3
#coding: utf8

from algorithm_3x import *
from threading import Thread 

debug = False

class Timer:
  def __init__(self, minutes, message):
    self._message = message
    self._minutes = minutes

  @staticmethod
  def calcRemainingTime(time_str):
    reg = re.compile(r"(\d+):(\d+)(am|pm)?", re.IGNORECASE)
    match = reg.findall(time_str.lower())
    if match == []:
      print("Error: -t time format is illegal")
      exit(1)
    match = match[0]
    hour, minutes, mark = int(match[0]), int(match[1]), match[2]
    if mark == "pm":
      hour += 12
    minutes = hour * 60 + minutes
    if debug:
      print("predefined time: %s hour %s minutes" %(hour, minutes))

    cur_time = time.localtime()
    cur_hour, cur_min = cur_time.tm_hour, cur_time.tm_min
    cur_minutes = cur_hour * 60 + cur_min
    if debug:
      print("currrent time: %s hour %s minutes" %(cur_hour, cur_min))

    if minutes <= cur_minutes:
      print("Error: -t time too early")
      exit(1)

    return minutes - cur_minutes

  @staticmethod
  def _acceptInput(status):
    while status[0]:
      input()
      status[0] = False

  def run(self):
    self._speak("开始倒计时%s分钟" %self._minutes)
    minutes = self._minutes
    #minutes = -1
    while minutes >= 0:
      print("remaining: %2s minutes\r" %minutes, end=' ')
      sys.stdout.flush()
      if minutes == 0:
        break
      if debug:
        time.sleep(1)
      else:
        time.sleep(60)
      minutes -= 1

    status = [True]
    td = Thread(target = Timer._acceptInput, args = (status,))
    td.start()

    for freq in range(9):
      self._speak("夏天先生你应该" + self._message + "了")
      time.sleep(3)
      if not status[0]:
        break

    td.join()

  def _speak(self, _message):
    cmd = "say -v Ting-Ting %s" %_message
    if debug:
      print(cmd)
    os.system(cmd)

if __name__ == "__main__":
  parser = optparse.OptionParser(usage = "cmd [10 minutes]")
  parser.add_option("-d", "--debug", action = "store_true", dest = "debug",
                    default = False, help = "debug mode")
  parser.add_option("-m", "--minutes", dest = "minutes", type = "int",
                    default = 0, help = "minutes")
  parser.add_option("-t", "--time", dest = "time", type = "string",
                    default = "", help = "example, 10:20am or 10:20 or 2:30pm")
  (options, args) = parser.parse_args()

  debug = options.debug
  if options.minutes > 0:
    minutes = options.minutes
  else:
    minutes = Timer.calcRemainingTime(options.time)

  message = "".join(args).strip()
  if message == "":
    message = "到时间"

  timer = Timer(minutes, message)
  timer.run()
