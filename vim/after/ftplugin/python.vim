""""""""""""""""""""""function definition""""""""""""""""""""""""""""""""""""""
function! GenOutline()

if !exists("b:open_fold")  
  let b:open_fold = 1
endif

python << endpython
import vim, re

open_fold = int(vim.eval("b:open_fold")) == 1
#print "open_fold:", open_fold
if not open_fold:
  vim.command("normal zE")
else:  
  buff = vim.current.buffer
  stack = []
  reg = re.compile(r"^\s*(class |def |@staticmethod)")

  for lineID, line in enumerate(buff):
    if reg.match(line) is not None:
      stack.append(lineID + 1)

  if stack != []:    
    if stack[0] != 1:
      stack = [1] + stack
    if stack[-1] != len(buff):
      stack.append(len(buff) + 1)

    for p in xrange(1, len(stack)):
      start, finish = stack[p - 1], stack[p] - 1
      if start < finish and not buff[start - 1].startswith("class"):
        vim.command(":%d, %d fold" %(start, finish))

endpython

let b:open_fold = !b:open_fold
"echo b:open_fold
endfunction

function! NewPython()
python << endpython
import vim

content = '''\
#!/usr/bin/env python3
#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

from palframe import nlp
from palframe import *

def main():
  parser = OptionParser(usage="cmd [optons] ..]")
  #parser.add_option("--quiet", action="store_true", dest="verbose",
                     #default=False, help="")
  (options, args) = parser.parse_args()

if __name__ == "__main__":
  main()
'''

buffer = vim.current.buffer
buffer[:] = content.split("\n")

endpython

:w % 
:!chmod 755 %
endfunction

""""""""""""""""""""""both for guivim and vim""""""""""""""""""""""""""""""""""
set softtabstop=2
set tabstop=2
set shiftwidth=2
set foldmethod=manual
set ignorecase

"call MapCodingBracket()
"inoremap {  {<CR>}<Esc>O
inoremap }   <C-R>=SuperEndMatch("}")<CR>
inoremap )   <C-R>=SuperEndMatch(")")<CR>
inoremap ]   <C-R>=SuperEndMatch("]")<CR>
inoremap "   <C-R>=SuperEndMatch('"')<CR>

map <F5>  :!./%<CR>
