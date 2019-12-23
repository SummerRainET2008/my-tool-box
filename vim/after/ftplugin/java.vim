""""""""""""""""""""""function definition""""""""""""""""""""""""""""""""""""""
function! SetBreakPoint(action)

python << endpython
import vim
import re

filename = vim.eval("expand('%')")
lineID = vim.eval("line('.')")
main_class = None

while True:
  vim.command("normal [{")
  clineID = vim.eval("line('.')")
  line = vim.eval("getline('.')")
  main_class = re.findall("[ ]?class\s* (\w+)", line)
  if clineID == lineID or main_class != []:
    break

package = ""
for line in vim.current.buffer:
  if line.startswith("import"):
    break
  if line.startswith("package"):
    package = re.findall("package\s* ([^ ]*?)\s*;", line)[0]

vim.current.window.cursor = (int(lineID), 0)
if main_class != []:
  action = vim.eval("a:action")
  main_class = main_class[0] if package == "" else package + "." + main_class[0]
  cmd = ":call VDBCommand('%s %s:%s')" %(action, main_class, lineID)
  vim.command(cmd)

  if action == "stop at":
    cmd = ":call VDBBreakSet(%s, '%s', %s)" %(lineID, filename, lineID) 
    vim.command(cmd)
  elif action == "clear": 
    cmd = ":call VDBBreakClear(%s, '%s')" %(lineID, filename) 
    vim.command(cmd)

endpython
endfunction

function! CompileJar()
python << endpython
import os

class_files = os.popen("find . -iregex '.*\.class'").read().split()
path = os.getcwd()
jar_name = os.path.split(path)[1].replace(" ", "-")
cmd = '''jar cvf %s.jar %s 1>/dev/null''' %(jar_name, " ".join(class_files)) 
#print cmd
os.system(cmd)

endpython
endfunction

function! ExtendNew()
python << endpython
import vim
import re 

line = vim.current.line
r1 =  re.findall(r"([\w<> ,]+)\s+\w+\s*=", line)
r2 =  re.findall(r"([\w<> ,]+)\[\]\s+\w+\s*=", line) 
r3 =  re.findall(r"([\w<> ,]+)\[\]\[\]\s+\w+\s*=", line) 
if r1 != [] and r1[0].strip() != "":
  ret = "new %s();" %r1[0].strip()
elif r2 != []: 
  ret = "new %s[];" %r2[0].strip()
elif r3 != []: 
  ret = "new %s[][];" %r3[0].strip()
else:  
  ret = "new"
ret = re.sub("<.*>", "<>", ret)  
vim.command("let ret = '%s'" %ret)

endpython

if ret == "new"
  return ret
else  
  return ret ."\<Left>\<Left>"
endif
endfunction
""""""""""""""""""""""both for guivim and vim""""""""""""""""""""""""""""""""""
"call MapCodingBracket()
inoremap {            <C-R>=SuperMatch()<CR>
inoremap }            <C-R>=SuperEndMatch("}")<CR>
inoremap )            <C-R>=SuperEndMatch(")")<CR>
inoremap ]            <C-R>=SuperEndMatch("]")<CR>
inoremap "            <C-R>=SuperEndMatch('"')<CR>

inoremap _print       System.out.println();<left><left>
inoremap _new         <C-R>=ExtendNew()<CR>
inoremap .            .<C-X><C-U>

