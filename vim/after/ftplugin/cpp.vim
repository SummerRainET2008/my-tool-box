""""""""""""""""""""""function definition""""""""""""""""""""""""""""""""""""""
function! AddMacroDefinitionForHeader()
python << endpython
import vim
  
flag_definition = '''\
#ifndef %s
#define %s

#endif

'''

fname = vim.current.buffer.name
buffer = vim.current.buffer
if fname.endswith(".h") and len(buffer) == 1 and len(buffer[0]) == 0:
  flag = fname.split("/")[-1].replace(".", "_").upper()
  buffer[:] = (flag_definition %(flag, flag)).split("\n")
else:
  print "The .h file must be empty"

endpython
endfunction

function! NewCplusplus()
python << endpython
import vim

content = '''\
#include "rain_algorithm.h"

using namespace rain;

int main() {
  cout << "Hello World" << endl;
 
  return 0;
}
'''

buffer = vim.current.buffer
buffer[:] = content.split("\n")

endpython
endfunction

function! ExecuteCplusplusProgram()
python << endpython
import vim
from os import listdir

found = False
if "BUILD" not in listdir("."):
  print "Not BUILD found"
else:
  '''binary = "test"'''
  for ln in open("BUILD"):
    ln = ln.strip()
    if ln.startswith("binary"):
      exe_file = ln.split("=")[1].strip()[1: -1]
      vim.command("silent !clear")
      vim.command("!./%s" %exe_file)
      break
  else:     
    print "Can not find any executable file"

endpython
endfunction

""""""""""""""""""""""both for guivim and vim""""""""""""""""""""""""""""""""""
" OmniCppComplete
let OmniCpp_NamespaceSearch = 1
let OmniCpp_GlobalScopeSearch = 1
let OmniCpp_ShowAccess = 1
let OmniCpp_MayCompleteDot = 1
let OmniCpp_MayCompleteArrow = 1
let OmniCpp_MayCompleteScope = 1
let OmniCpp_DefaultNamespaces = ["std", "_GLIBCXX_STD"]

call omni#cpp#complete#Init()

"call MapCodingBracket()
inoremap {   <C-R>=SuperMatch()<CR>
inoremap }   <C-R>=SuperEndMatch("}")<CR>
inoremap )   <C-R>=SuperEndMatch(")")<CR>
inoremap ]   <C-R>=SuperEndMatch("]")<CR>
inoremap "   <C-R>=SuperEndMatch('"')<CR>

map <C-b>         :!_make.py<CR>
map <C-b><C-c>    :!_make.py -c<CR>

map <C-F2>        :!ctags --exclude="excluded*" -R --c++-kinds=+defgpstux --fields=+iaSKlnz --extra=+q .<CR><CR>

map <F5>          :call ExecuteCplusplusProgram()<CR>

