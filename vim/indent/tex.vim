if exists("b:did_indent")
  finish
endif
let b:did_indent = 1

setlocal nolisp         " Make sure lisp indenting doesn't supersede us
setlocal autoindent     " indentexpr isn't much help otherwise

setlocal indentexpr=TexIndent(v:lnum)
setlocal indentkeys=!^F,o,O,<:>,0),0],0}

function! TexIndent(lnum)

if a:lnum == 1
  return 0
endif

let prev_num = prevnonblank(a:lnum - 1)
let prev_line = getline(prev_num)
let prev_indent = indent(prev_num)
let line = getline(a:lnum)

if line =~ '^\s*\\end{' || line =~ '^\s*}\s*$'
  return prev_indent - 2
elseif prev_line =~ '^\s*\\begin' || prev_line =~ '{\s*$' 
  return prev_indent + 2
else
  return prev_indent
endif

endfunction
