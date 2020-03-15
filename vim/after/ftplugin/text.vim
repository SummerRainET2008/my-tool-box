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
  reg = re.compile(r"^(\d+\.|---|!\.)")

  for lineID, line in enumerate(buff):
    if reg.match(line) is not None:
      stack.append(lineID + 1)

  if stack != []:    
    if stack[0] != 1:
      stack = [1] + stack
    if stack[-1] != len(buff):
      stack.append(len(buff) + 1)

    for p in xrange(1, len(stack)):
      vim.command(":%d, %d fold" %(stack[p - 1], stack[p] - 1))

endpython

let b:open_fold = !b:open_fold
"echo b:open_fold
endfunction

function! GenEnumerationIndex()
python << endpython
import vim, re

reg = re.compile(r'''^((\s*)\d+)[.)]''')
indentStack = [[-1, -1]]     # [indent size, idx]
buff = vim.current.buffer 

line_ID = 0
while line_ID < len(buff):
  ln = buff[line_ID]
  if ln.startswith("---") or ln.endswith("---") or ln.startswith("!."):
    indentStack = [[-1, -1]]
    line_ID += 1
    continue

  match = reg.findall(ln)
  if match == []:
    line_ID += 1
  else:
    match = match[0]
    #print "line:", line_ID, "match:", match 
    indentSize = len(match[1])
    if indentSize < indentStack[-1][0]:
      indentStack.pop()
      continue

    if indentSize > indentStack[-1][0]:
      indentStack.append([indentSize, 1])
    else:
      indentStack[-1][1] += 1

    nstr = " " * indentStack[-1][0] + str(indentStack[-1][1])
    prefix = match[0]
    buff[line_ID] = nstr + ln[len(prefix):] 
    line_ID += 1

endpython
endfunction

function! SetTitle()

let line_id = getpos(".")[1] - 1

python << endpython
import vim, re

line_id = int(vim.eval("line_id"))
line = vim.current.buffer[line_id] 
max_len = 80
blank_num = max_len - (len(line) + 2)
title_line = "-" *  (blank_num/ 2) + " " + line + " " +  "-" * (blank_num - blank_num / 2)
vim.current.buffer[line_id] = title_line

endpython
endfunction  

set foldmethod=manual
set tw=0
set wrap
map --- :call SetTitle()<CR> 

syn match Define          "^---.*"
syn match Define          ".*---$"
syn match String          "^\d\+\..*"
syn match String          "^!\..*"

map <F3>        :call GenOutline()<CR>
map <F5>        :call GenEnumerationIndex()<CR>

call ConcelLink() 
