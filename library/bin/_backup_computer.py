#!/usr/bin/env python3

from palframe import nlp
import optparse
import os
import time

def main():  
  parser = optparse.OptionParser(usage = "cmd [optons]")
  parser.add_option("-d", action="store_true", dest="delete",
                    help="to delete additional files.")
  parser.add_option("--server", default="ubuntu_wifi",
                    help="default 'ubuntu_wifi'")
  (options, args) = parser.parse_args()

  src_path = os.path.expanduser("~/inf")
  assert os.path.isdir(src_path)
  os.chdir(src_path)
  src_path = "."

  target_path = f"{options.server}@/media/ubuntu/backup/inf"

  cmd = f"_supdate.py {src_path} {target_path}"
  if options.delete:
    cmd += " -d"

  print("--" * 64)
  print(f"to delete   : {options.delete}")
  print()

  print(f"pwd: {os.getcwd()}")
  print(f"cmd to excute: {cmd}")

  print("--" * 64, "\n")

  answer = input("continue [y | n] ? >> ")
  if answer == "y":
    start_time = time.time()

    nlp.execute_cmd(f"chmod -Rv a+r *")
    nlp.execute_cmd("find . -type d -exec chmod -v a+rx {} \;")

    nlp.execute_cmd(cmd)

    nlp.execute_cmd("_clone_file_tag.py --cmd gen")

    duration = time.time() - start_time
    print(f"time: {duration} seconds.")

if __name__ == "__main__":
  main()
