let s:connected   = 0

highlight DebugBreak guibg=darkred    guifg=white ctermbg=darkred    ctermfg=white
highlight DebugStop  guibg=darkblue   guifg=white ctermbg=darkblue   ctermfg=white

sign define breakpoint linehl=DebugBreak
sign define current    linehl=DebugStop

function! VDBInit(fifo, pwd)
  if s:connected == 1
    echo "Already communicating with a VDB Session!"
    return
  endif

  let s:connected = 1
  let s:fifo = a:fifo
  execute "cd " . escape(a:pwd, " ")
  
  if !exists(":Vdb")
    command -nargs=+ Vdb        :call VDBCommand(<q-args>, v:count)
  endif

  python import os
  silent exec 'python os.chdir("' . a:pwd . '")'
  silent exec 'python fifo = "' . s:fifo . '"'
endfunction

function! VDBClose()
  sign unplace *
  let s:connected = 0
endfunction

function! VDBCommand(cmd, ...)
  if match (a:cmd, '^\s*$') != -1
    return
  endif
  
  " Create command arguments
  let suff=""
  if 0 < a:0 && a:1 != 0
    let suff = " " . a:1
  endif
  
  " Send the command
  python fd = open(fifo, "w")
  exec 'python fd.write("%s\n" % "' . a:cmd . suff . '")'
  python fd.close()
endfunction

function! VDBBreakSet(id, file, linenum)
  call VDBJumpToLine(a:linenum, a:file)
  execute "sign unplace " . a:id
  execute "sign place   " . a:id . " name=breakpoint line=".a:linenum." file=" . @%
endfunction

function! VDBBreakClear(id, file)
  execute "sign unplace " . a:id . " file=".a:file
endfunction

function! VDBJumpToLine(line, file)
  if !bufexists(a:file)
      "echom "!bufexists " . a:line
      execute "e ". escape(a:file, " ")
  else
      "echom "bufexists " . a:line
      execute "b ". escape(a:file, " ") 
  endif
  execute a:line
  :silent! foldopen!
endfunction

function! VDBHighlightLine(line, file)
  call VDBJumpToLine(a:line, a:file)
  execute "sign unplace ". 1
  execute "sign place " .  1 ." name=current line=" . a:line . " file=" . @%
endfunction


