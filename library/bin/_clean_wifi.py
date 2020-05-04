#!/usr/bin/env python3
#coding: utf8

from pa_nlp import common as nlp
import optparse
import os

if __name__ == "__main__":
  parser = optparse.OptionParser(usage = "cmd server 'cmd'")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  (options, args) = parser.parse_args()

  os.chdir("/Library/Preferences/SystemConfiguration")
  nlp.execute_cmd("sudo rm -v com.apple.airport.preferences.plist")
  nlp.execute_cmd("sudo rm -v NetworkInterfaces.plist")
  nlp.execute_cmd("sudo rm -v preferences.plist") 
