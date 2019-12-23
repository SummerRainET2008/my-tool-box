from sys import argv

def is_zh_char(uchar):
    '''check a unicode char is chinese'''
    return uchar >= '\u4e00' and uchar <= '\u9fa5'

def char_B2Q(uchar):
    '''char of B2Q'''
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e:   
        return uchar
    inside_code = 0x3000 if inside_code == 0x0020 else inside_code + 0xfee0
    return chr(inside_code)

def char_Q2B(uchar):
    '''char of Q2B'''
    inside_code = ord(uchar)
    inside_code = 0x0020 if inside_code == 0x3000 else inside_code - 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e: 
        return uchar
    return chr(inside_code)

def str_Q2B(ustr):
    '''transfor a unicode string from Q 2 B'''
    return "".join([char_Q2B(uchar) for uchar in ustr])

def str_B2Q(ustr):
    '''transfor a unicode string from Q 2 B'''
    return "".join([char_B2Q(uchar) for uchar in ustr])

def count_zh(ustr):
    return len([c for c in ustr if is_zh_char(c)])

if __name__ == "__main__":
    assert len(argv) == 4, "cmd file.in (quan|ban) code(gbk|utf8)"

    code = argv[3]
    if argv[2] == "quan":
        print(str_B2Q(open(argv[1]).read().decode(code)).encode(code), file=open(argv[1] + ".quan", "w"))
    elif argv[2] == "ban":
        print(str_Q2B(open(argv[1]).read().decode(code)).encode(code), file=open(argv[1] + ".ban", "w"))
    else:
        print("quan or ban, input error!")
