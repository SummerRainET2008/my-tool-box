#!/usr/bin/env python
 
from algorithm import * 
import re
import urllib.request, urllib.parse, urllib.error
import mechanize

expect_suffix = ["pdf", "doc", "ppt", "ps", "docx"]

def analyze_links(url):
    '''visit http://stockrt.github.com/p/emulating-a-browser-in-python-with-mechanize/
    for more ways to set up your br browser object e.g. so it look like mozilla
    and if you need to fill out forms with passwords.'''
    br = mechanize.Browser()
    br.open(url)

    #you can also iterate through br.forms() to print forms on the page!
    for lk in br.links(): 
        for sf in expect_suffix: 
            if lk.url.endswith(sf):
                yield lk.text, lk.absolute_url

def download(url):
    return urllib.request.urlopen(url).read()

def shortname(url):
    return url.split("/")[-1]

def linktype(url):
    return url.split(".")[-1]

if __name__ == "__main__":
    parser = OptionParser(usage = "cmd -u url")
    parser.add_option("-u", "--url", dest = "url", help = "url of website")
    (options, args) = parser.parse_args()

    print(download(options.url), file=open("mainpage.html", "w")) 
    print(options.url, file=open("url.txt", "w")) 

    for ind, (text, path) in enumerate(analyze_links(options.url)):
        print("%3d-th link, text: '%s', path='%s'" %(ind, text, path))
        print()
        name = "%s (%s).%s" %(text, shortname(path), linktype(path))
        try:
            cont = download(path)
            print(cont, file=open(name, "w")) 
        except IOError:
            print("find an error, and ignore:", name)

