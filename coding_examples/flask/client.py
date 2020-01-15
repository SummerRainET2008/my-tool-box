import urllib.parse
import urllib.request
import json

if __name__ == '__main__':
  text = urllib.parse.quote("summer rain hahah why yun")
  # print(os.system(f"curl -i  'http://192.168.1.201:5000/sim/tasks/{text}'"))
  url = f"http://localhost:5000/sim/tasks/{text}"
  s = urllib.request.urlopen(url).read()
  print(json.loads(s))

