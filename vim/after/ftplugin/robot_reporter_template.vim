""""""""""""""""""""""function definition""""""""""""""""""""""""""""""""""""""
function! LoadTemplateVar(keyword)

"list[f: t] ---> [f, t], not [f, t) in Python.
let pattern = a:keyword[: -2] . '(:\d+)?$'
let pattern = escape(pattern, '$(+)?.')
let cmd = printf("syn match Define '%s'", pattern)
execute cmd

endfunction

function! LoadAllTemplateVar()
python << endpython
import vim

try:
  VIMHOME = vim.eval("g:VIMHOME")
  fname = "%s/data/robot_reporter_template_variables.tpt" %VIMHOME
  variables = open(fname).read().split()
  for v in variables:
    vim.command('''call LoadTemplateVar("%s")''' %v)
except IOError:
  print "Can not find '%s'" %fname

endpython
endfunction  

""""""""""""""""""""""both for guivim and vim""""""""""""""""""""""""""""""""""
if exists("b:current_syntax")
  finish
endif

let b:current_syntax = "robot_reporter_template"
scriptencoding utf-8
setlocal iskeyword+=.

syn match Todo "[-+]\?\d\+"
syn match Todo "[-+]\?\d\+\.\d\+"
syn match Todo "[一二三四五六七八九十]"
"hi def link  Number Special

syn match Todo "减少"
syn match Todo "下降"
syn match Todo "流出"

syn match Label ":template"
syn match Label ":paragraph"
syn match Label ":if"
syn match Label ":else"
syn match Label ":random_one"
syn match Label ":random_order"
syn match Label ":sequence"
syn match Label ":title"
syn match Label ":section"
syn match Label ":chart"
syn match Label ":table"
syn match Label ":text"

call LoadAllTemplateVar()

call MapCodingBracket()
inoremap {  {<CR>}<Esc>O
inoremap }   <C-R>=SuperEndMatch("}")<CR>
inoremap )   <C-R>=SuperEndMatch(")")<CR>
inoremap ]   <C-R>=SuperEndMatch("]")<CR>
