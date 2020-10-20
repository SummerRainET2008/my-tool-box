#!/usr/bin/env python3

import optparse
from PyPDF2 import PdfFileMerger

def pdf_cat(input_files, out_file):
  try:
    merger = PdfFileMerger()
    for pdf in input_files:
      merger.append(open(pdf, "rb"))

    merger.write(open(out_file, "wb"))
    merger.close()
    return True

  except Exception as error:
    # print(input_files, error)
    return False

def main():
  parser = optparse.OptionParser(usage = "cmd [optons]")
  parser.add_option("--out_file", default="out.pdf")
  (options, args) = parser.parse_args()

  valid_files = []
  for f in args:
    if pdf_cat([f], "/tmp/valid.pdf"):
      valid_files.append(f)
    else:
      print("ERR:", f)

  result = pdf_cat(valid_files, options.out_file)
  assert result
  print(f"Combined {valid_files}")

if __name__ == '__main__':
  main()
