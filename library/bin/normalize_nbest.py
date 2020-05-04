#!/usr/bin/python

'''
    summer rain!
'''

from sys import argv
from re import compile
import optparse

regs = [(compile("<skipped>"), ""),
        (compile("&quot;"), '"'),
        (compile("&amp;"), "&"),
        (compile("&lt;"), "<"),                                 # convert SGML tag for less-than to >
        (compile("&gt;"), ">"),                                 # convert SGML tag for greater-than to <
        (compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),  # tokenize punctuation
        (compile(r"([^0-9])([\.,])"), r"\1 \2 "),               # tokenize period and comma unless preceded by a digit
        (compile(r"([\.,])([^0-9])"), r" \1 \2"),               # tokenize period and comma unless followed by a digit
        (compile(r"([0-9])(-)"), r"\1 \2 "),                    # tokenize dash when preceded by a digit
        (compile(r"\s+"), " "),                                 # one space only between words
        (compile(r"([a-zA-Z]+) - ([a-zA-Z]+)"), r"\1-\2"),
        (compile(r"cann't"), "can't"),
        (compile(r"wonn't"), "won't"),
        (compile(r"(i|we|she|he|you|they) '"), r"\1'")]
fmt_reg = compile(r"(<seg.*?>)(.*?)(</seg>)")

def exist_tag(fe):
    fin = file(fe, "rU")
    fin.next()
    tag = fin.next().strip()
    return not tag.isdigit()

def normalize(txt):
    for reg, s in regs:
        txt = reg.sub(s, txt)
    return txt        

if __name__ == "__main__":
    optparser = optparse.OptionParser()
    optparser.add_option("-l", "--lower", action="store_true", default=False, dest = "lowercase", help = "lowercase")
    (opts, args) = optparser.parse_args()

    for fe in args:
        if opts.lowercase:
            fou = file("%s.norm.lower" %fe, "w")
        else:
            fou = file("%s.norm" %fe, "w")
        fin = file(fe, "rU") 
        istag = exist_tag(fe)
        src_num = int(fin.next())
        print >> fou, src_num
        for src_id in xrange(src_num):
            if istag:
                print >> fou, fin.next(),
            nhyp = int(fin.next())
            print >> fou, nhyp
            for hid in xrange(nhyp):
                hyp, _, feat = fin.next().strip().partition("|||")
                if opts.lowercase:
                    hyp = hyp.lower()
                hyp = normalize(hyp)
                print >> fou, "%s ||| %s" %(hyp, feat)
        fou.close()                
        print fe, "is OK"


