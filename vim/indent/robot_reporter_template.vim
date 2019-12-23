" Vim indent file
" Language:             robot_reporter_template.vim

" Only load this indent file when no other was loaded.
if exists("b:did_indent")
  finish
endif
let b:did_indent = 1

" Some preliminary settings
setlocal nolisp         " Make sure lisp indenting doesn't supersede us
setlocal autoindent     " indentexpr isn't much help otherwise

setlocal indentexpr=RobotIndent(v:lnum)
setlocal indentkeys=!^F,o,O,<:>,0),0],0}

function! RobotIndent(lnum)
  " lnum, indent, prevnonblank, and many functions start with 1, rather than 0.
  " But its built-in array starts with 0.
  " :let mylist = [1, 2, ['a', 'b']]
  " :echo mylist[0]

  if a:lnum == 1
    return 0
  endif

  let prev_line_idx = prevnonblank(a:lnum - 1)
  let prev_line = getline(prev_line_idx)
  let prev_indent = indent(prev_line_idx)
  let cur_line = getline(a:lnum)

  if cur_line =~ '[}\]]\s*$'
    return prev_indent - 2
  elseif prev_line =~ '[{\[]\s*$'
    return prev_indent + 2
  else
    return prev_indent
  endif

endfunction

