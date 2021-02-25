#!/usr/bin/env python

# http://www.360doc.com/content/12/0218/12/4920176_187558951.shtml

from palframe.nlp import *
from datetime import date 
from math import sin
import pylab  

days_circle_intellegence = 33
days_circle_mood         = 28
days_circle_strength     = 23

def myrange(f, t, step):
  while f < t:
    yield f
    f += step

def draw_circle(figure_id, circle, mod, label):
  step = 0.2
  pylab.figure(figure_id)

  # draw x-line
  x = list(myrange(0, circle + step, step))
  y = [0] * len(x)
  pylab.plot(x, y, "-")

  # draw sin()
  pi = 3.1415926
  x = list(myrange(0., circle + step, step))
  y = [16 * sin(2 * pi * _x / circle) for _x in x]
  pylab.plot(x, y, "o", label = label)

  # my my status
  y = list(myrange(-18 - step, 18 + step, step))
  x = [mod] * len(y)
  pylab.plot(x, y, "^")
 
  pylab.grid()
  pylab.legend()

if __name__ == "__main__":
  parser = optparse.OptionParser(usage = "cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",\
      #default = False, help = "don't print status messages to stdout")
  parser.add_option("-b", "--birthday", dest = "birthday",\
      default = "1984 11 9", help = "default: 1984 11 9")
  (options, args) = parser.parse_args()

  year, month, day = options.birthday.split()
  birth_date = date(int(year), int(month), int(day))
  days = birth_date.today().toordinal() - birth_date.toordinal()

  mod_intellegence = days % days_circle_intellegence
  mod_mood = days % days_circle_mood
  mod_strength = days % days_circle_strength

  #print "intellegence (%d): %d" %(days_circle_intellegence, mod_intellegence)
  #print "mood (%d): %d" %(days_circle_mood, mod_mood)
  #print "strength (%d): %d" %(days_circle_strength, mod_strength)

  draw_circle(0, days_circle_intellegence, mod_intellegence, "Intellegence")
  draw_circle(1, days_circle_mood, mod_mood, "Mood")
  draw_circle(2, days_circle_strength, mod_strength, "Strength")
  pylab.show()

  

