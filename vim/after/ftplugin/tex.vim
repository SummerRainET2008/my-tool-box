""""""""""""""""""""""function definition""""""""""""""""""""""""""""""""""""""
function! GenLatexTags()
python << endpython
import vim, re

labels = {}
label_heads = ["fig", "eq", "tb", "thm", "sec", "alg"]
for head in label_heads:
  labels.setdefault(head, [])

reg = re.compile(r"\label{(.*?)}")
tags = reg.findall(" ".join(vim.current.buffer))
for tag in tags:
  for head in labels:
    if tag.startswith(head):
      labels[head].append(tag)
      break
  else:    
    print "Unknow labels:", tag
    print "label must start with any of '%s'" %", ".join(label_heads)

try:
  txt = open("references.bib", "r").read() 
  reg = re.compile(r"^@.*?{(.*?),", re.MULTILINE)
  labels["citation"] = reg.findall(txt)
except:
  print "Warning: No references.bib"

vim.command("let w:latex_labels = %s" %labels) 
  
endpython
endfunction

function! InsertLatexLabel()
  if !exists("w:latex_labels")
    call GenLatexTags()
  endif

  let context = strpart(getline("."), 0, col(".") - 2)
  "echom "context: " . context
  if context =~ '.*Figure\s* \\ref'  
    call complete(col('.'), w:latex_labels["fig"])
  elseif context =~ '.*Eqn\.\s* \\ref'  
    call complete(col('.'), w:latex_labels["eq"])
  elseif context =~ '.*Table\s* \\ref'
    call complete(col('.'), w:latex_labels["tb"])
  elseif context =~ '.*Theorem\s* \\ref'
    call complete(col('.'), w:latex_labels["thm"])
  elseif context =~ '.*Alg\.\s* \\ref'
    call complete(col('.'), w:latex_labels["alg"])
  elseif context =~ '.*\\cite'   
    call complete(col('.'), w:latex_labels["citation"])
  endif

  return ''
endfunction

function! CompileLatex(pre_compile)
python << endpython
import vim

file_type = [
  "aux", 
  "blg", 
  "brf",
  "idx",
  "loa",
  "lof",
  "log", 
  "lot",
  "out",
  "toc",
  "pdfsync",
]

def latex_clean(fn, file_type_to_delete):
  for suffix in file_type_to_delete:
    if suffix == "pdf":
      cmd = '''silent !rm "%s.%s"''' %(fn, suffix)
    else:
      cmd = '''silent !rm *.%s''' %(suffix)
    vim.command(cmd)

file_name = vim.current.buffer.name.split("/")[-1]
if not file_name.endswith(".tex"):
  print "Does not end with .tex"
else: 
  file_name = file_name.replace(".tex", "")

  cmd0 = '''silent !pdflatex "%s"''' %file_name
  cmd1 = '''!pdflatex "%s"''' %file_name
  cmd2 = '''silent !bibtex "%s"''' %file_name
  cmd3 = '''!open "%s.pdf"''' %file_name
 
  pre_compile = vim.eval("a:pre_compile") == "1"
  latex_clean(file_name, file_type + ["pdf"])   
  if pre_compile:
    vim.command(cmd0) 
    vim.command(cmd3)
  else:
    vim.command(cmd0) 
    vim.command(cmd2) 
    vim.command(cmd0) 
    vim.command(cmd0) 
    vim.command(cmd3)
  latex_clean(file_name, file_type)   

endpython
endfunction

""""""""""""""""""""""both for guivim and vim""""""""""""""""""""""""""""""""""
setlocal iskeyword+=_
setlocal textwidth=0
set wrap

call MapCodingBracket()
inoremap {  {}<Left>
inoremap }   <C-R>=SuperEndMatch("}")<CR>
inoremap )   <C-R>=SuperEndMatch(")")<CR>
inoremap ]   <C-R>=SuperEndMatch("]")<CR>

" Does not convert any math symbols in latex.
let g:tex_conceal=""

" note, <C-b> = <C-B>; however, b != B generally.
map <C-b>         :call CompileLatex(1)<CR>
map <S-b>         :call CompileLatex(0)<CR>

inoremap <F5>     <C-R>=InsertLatexLabel()<CR>
map <F6>          :call GenLatexTags()<CR>

map <F7>          :call MatchNoneAsciiChars()<CR>
