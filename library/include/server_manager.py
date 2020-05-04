#!/usr/bin/env python3
#coding: utf8

from pa_nlp import nlp
import optparse

class ServerManager:
  _inst = None

  def __init__(self):
    self._servers = {}
    home = nlp.get_home_dir()
    servers = nlp.pydict_file_read(f"{home}/my-tool-box/"
                                   f"library/data/servers.pydict")
    for server in servers:
      self._servers[server["name"]] = server
      print(server)

  def get_port(self, server_name):
    server = self._servers[server_name]
    return server.get("port", "")

  def get_ip(self, serverName):
    if serverName == "localhost":
      return serverName
    return self._servers.get(serverName, {"ip": None})["ip"]

  def get_login(self, serverName):
    if serverName not in self._servers:
      return None

    server = self._servers[serverName]
    return "%s@%s" %(server["account"], server["ip"])

  @staticmethod
  def get_instance():
    if ServerManager._inst is None:
      ServerManager._inst = ServerManager()

    return ServerManager._inst    

if __name__ == "__main__":
  parser = optparse.OptionParser(usage = "cmd dev1@dir1 dir2")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  (options, args) = parser.parse_args()

  serverManager = ServerManager.get_instance()
