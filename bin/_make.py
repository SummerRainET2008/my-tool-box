#!/usr/bin/env python3

import os
from os import path, system
from optparse import OptionParser
from multiprocessing import Pool

BUILD_format = '''
binary = "hello-world"
srcs   = [
  "src1.cpp",
]

cplusplus_standard = "c++11"  # candidates "c++98"
                              # default: c++11

include_paths = [
  "path1",
  "path2",
]
compile_flags = [""]

lib_paths = [
  "path1",
  "path2",
]
libs = [
  "file1.so",
  "file2.so",
]
link_flags = ["-static-libgcc"]
  '''

def thread_compile(args):
  cmd, error_file = args
  fname     = error_file.replace(".error", "")[1:]
  exec_code = system(cmd)
  if exec_code != 0:
    return [fname, open(error_file).readlines()] 
  else:
    error_txt = open(error_file).read().strip()
    if error_txt != "":
      print("-" * 16, "Warning", "-" * 16)
      print(error_txt)
      print()
    return [fname, None]

def get_default_config():
  default_config = {
    "cplusplus_standard":   "c++11",
    "include_paths":        [],
    "compile_flags":        ["-fPIC", "-Wall", "-Wextra"],
    "lib_paths":            [],
    "libs":                 [],
    "link_flags":           [],
  }
  return default_config

def combine_config(user_config, default_config):
  config = {}

  config["binary"] = user_config["binary"]
  config["srcs"]   = user_config["srcs"]
  config["cplusplus_standard"] = user_config.get(
    "cplusplus_standard", default_config["cplusplus_standard"]
  )
  config["include_paths"] = default_config["include_paths"] \
      + user_config.get("include_paths", [])
  config["compile_flags"] = default_config["compile_flags"] \
      + user_config.get("compile_flags", [])
  config["lib_paths"]        = default_config["lib_paths"] \
      + user_config.get("lib_paths", [])
  config["libs"]       = default_config["libs"] \
      + user_config.get("libs", [])    
  config["link_flags"]  = default_config["link_flags"] \
      + user_config.get("link_flags", [])   
  return config

def get_list_to_str(config, attr, prefix):
  return " ".join(f"{prefix}{item}" for item in config[attr])

def gen_hidden_file(full_name: str):
  file_path, base_name = os.path.split(full_name)
  assert base_name != "", full_name
  if file_path != "":
    return f"{file_path}/.{base_name}"
  return f".{base_name}"

def change_cpp_file_extention(file_name: str, target_ext: str):
  assert target_ext.startswith(".")
  return file_name\
    .replace(".cpp", target_ext).replace(".cc", target_ext)

if __name__ == "__main__":
  #parser = OptionParser(usage = "cmd [-t compile-type]")
  parser = OptionParser(usage = "cmd [-t compile-type]\n" + 
      "BUILD format" + BUILD_format)
  parser.add_option("-r", "--release", dest = "release",
      action = "store_true", default = False)
  parser.add_option("-p", "--profile", dest = "profile", 
      action = "store_true", default = False)
  parser.add_option("-d", "--debug",   dest = "debug",   
      action = "store_true", default = True)
  parser.add_option("-c", "--clean",   dest = "clean",   
      action = "store_true", default = False)
  parser.add_option("-n", "--new_BUILD",   dest = "new_BUILD",   
      action = "store_true", default = False)

  (options, args) = parser.parse_args()

  if options.new_BUILD:
    print(BUILD_format, file=open("BUILD", "w"))
    exit(0)

  try:
    user_config = {}
    exec(compile(open("BUILD").read(), "BUILD", 'exec'), user_config) 
  except IOError:
    print("Can NOT find BUILD file")
    exit(1)

  config = combine_config(user_config, get_default_config()) 

  ERROR_FILE  = ".et.errors.cpp"
  system("rm %s; rm %s; clear" %(ERROR_FILE, config["binary"]));

  if options.release:
    binary_type = "release"
    config["compile_flags"].append("-O3")
  elif options.profile:
    binary_type = "profile"
    config["compile_flags"].append("-pg")
    config["link_flags"].append("-pg")
  elif options.debug:
    config["compile_flags"].append("-g")
    binary_type = "debug"

  if config["cplusplus_standard"] == "c++11":
    config["compile_flags"].append("-std=c++11")

  src_files = config["srcs"] 
  cpp_to_compile = []
  dot_o_files = []

  for src_file in src_files:
    print(f"{'-' * 20} {src_file} {'-' * 20}")
    dot_o_file = gen_hidden_file(change_cpp_file_extention(src_file, ".o"))
    error_file = gen_hidden_file(change_cpp_file_extention(src_file, ".error"))

    time1 = path.getmtime(src_file)
    if path.isfile(dot_o_file):
      time2 = path.getmtime(dot_o_file)
    else:
      time2 = time1 - 1 

    if time1 > time2 or options.clean:
      include_paths_str = get_list_to_str(config, "include_paths", "-I") 
      compile_flags_str = get_list_to_str(config, "compile_flags", "")

      cmd = f"g++ -c {src_file} {compile_flags_str} -o {dot_o_file} " \
        f"{include_paths_str} 2> {error_file}"
      print(cmd)
      cpp_to_compile.append((cmd, error_file))
    else:
      print(f"{dot_o_file} is the newest")
    dot_o_files.append(dot_o_file)    

  thread_pools = Pool()
  errors = thread_pools.map(thread_compile, cpp_to_compile)
  error_inf = []
  for cpp, error in errors:
    if error is not None:
      error_inf.extend(error)
  
  if error_inf != []:
    print("".join(error_inf), file=open(ERROR_FILE, "w"))
    print("".join(error_inf[: 25]))
  else: 
    print("-" * 48)
    link_flags_str = get_list_to_str(config, "link_flags", "")
    lib_paths_str  = get_list_to_str(config, "lib_paths", "-L")
    libs_str       = get_list_to_str(config, "libs", "-l")

    cmd = f"g++ {' '.join(dot_o_files)} -o {config['binary']} " \
      f"{link_flags_str} {libs_str} {lib_paths_str}"
    print(cmd)

    if system(f"{cmd} 2> {ERROR_FILE}") == 0:
      print(f"Successful! binary type: {binary_type}, {config['binary']}")
    else:
      print("".join(open(ERROR_FILE).readlines()[: 20]))

