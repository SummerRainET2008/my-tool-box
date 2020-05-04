#include <cassert>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

typedef string::size_type size_type;

bool g_Verbose = false;
vector<bool> g_vConvertFlag;

//void hzj(string& s)
//{
//    char halfbuf[1000000];
//    int j = 0;
//
//    unsigned char c1, c2;
//
//    for(int i = 0; i < s.size(); i++)
//    {
//        c1 = (unsigned char)s[i];
//        c2 = (unsigned char)s[i+1];
//
//        if(c1 == 163) // SBC case
//        {
//            if((c2-128) == 10 || (c2-128) == 13) //回车 换行
//            {
//                i++;
//                continue;
//            }
//            halfbuf[j++] = char(c2-128);
//            i++;
//            continue;
//        }
//        else if(c1 == 161 && c2 == 161)//全角空格
//        {
//            halfbuf[j++] = ' ';
//            i++;
//            continue;
//        }
//        else if(c1 ==161 &&  c2==164)
//        {
//            halfbuf[j++]='.';
//               i++;
//               continue;
//        }
//        else if(c1 &0X80) // Chinese characters
//        {
//            halfbuf[j++] = char(c1);
//            halfbuf[j++] = char(c2);
//            i++;
//            continue;
//        }
//        else
//        {
//            halfbuf[j++] = char(c1);
//        }
//    }
//
//    halfbuf[j] = 0;
//    string result(halfbuf);
//    s=result;
//}

///////////////////

string full2half_A3(const string& full) {
  size_type i;
  string half;
  for (i = 0; i < full.length(); ) {
    char c1 = full[i];
    char c2 = full[i + 1];
    bool flag = (c1 < 0);
    //  space
    if (c1 == '\xA1' && c2 == '\xA1' && g_vConvertFlag[0]) {
      //cerr<<"c1=A1 and c2=A1"<<endl;
      half += ' ';
    }
    //  a3 section
    else
      if (c1 == '\xA3' && c2 > '\xA0' && c2 < '\xFF' && g_vConvertFlag[c2 - '\xA0']) {
        half += c2 + 128;
      }
      else
        if (flag) {
          half += c1;
          half += c2;
        }
        else {
          half += c1;
        }
    //  move next i
    if (flag) {
      i += 2;
    }
    else {
      ++i;
    }
  }
  return half;
}

void Usage(int argc, char* argv[]) {
  cerr << "Usage:" << endl;
  cerr << "    " << argv[0] << " [OPTION]" << endl;
  cerr << "Function:" << endl;
  cerr << "    Convert SBC(full) case to DBC(half) case. " << endl;
  cerr << "    Read from standard input and write to standard output." << endl;
  cerr << "    Only convert characters in section A3." << endl;
  cerr << "Option:" << endl;
  cerr << "    -c, --character    Convert \'A\'-\'Z\' and \'a\'-\'z\'." << endl;
  cerr << "    -u, --upper        Convert \'A\'-\'Z\'." << endl;
  cerr << "    -l, --lower        Convert \'a\'-\'z\'." << endl;
  cerr << "    -p, --punction     Convert punction." << endl;
  cerr << "    -s, --space        Convert space." << endl;
  cerr << "    -d, --digit        Convert \'0\'-\'9\'." << endl;
  cerr << "    -a, --all          Convert all characters in section A3." << endl;
  cerr << "    -h, --help         Display this messege and exit." << endl;
  cerr << "    -v, --verbose      Verbose mode." << endl;
  exit(1);
}

void InitOption(int argc, char* argv[]) {
  g_vConvertFlag.resize(96, false);
  int i;
  for (i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      Usage(argc, argv);
    }
    else
      if (arg == "-v" || arg == "--verbose") {
        g_Verbose = true;
      }
      else
        if (arg == "-c" || arg == "--character") {
          fill_n(g_vConvertFlag.begin() + ('A' - ' '), 26, true);
          fill_n(g_vConvertFlag.begin() + ('a' - ' '), 26, true);
        }
        else
          if (arg == "-u" || arg == "--upper") {
            fill_n(g_vConvertFlag.begin() + ('A' - ' '), 26, true);
          }
          else
            if (arg == "-l" || arg == "--lower") {
              fill_n(g_vConvertFlag.begin() + ('a' - ' '), 26, true);
            }
            else
              if (arg == "-p" || arg == "--punction") {
                fill(g_vConvertFlag.begin() + ('!' - ' '), g_vConvertFlag.begin() + ('0' - ' '),
                     true);
                fill(g_vConvertFlag.begin() + (':' - ' '), g_vConvertFlag.begin() + ('A' - ' '),
                     true);
                fill(g_vConvertFlag.begin() + ('[' - ' '), g_vConvertFlag.begin() + ('a' - ' '),
                     true);
                fill(g_vConvertFlag.begin() + ('{' - ' '), g_vConvertFlag.end(), true);
              }
              else
                if (arg == "-s" || arg == "--space") {
                  g_vConvertFlag[' ' - ' '] = true;
                }
                else
                  if (arg == "-d" || arg == "--digit") {
                    fill_n(g_vConvertFlag.begin() + ('0' - ' '), 10, true);
                  }
                  else
                    if (arg == "-a" || arg == "--all") {
                      fill(g_vConvertFlag.begin(), g_vConvertFlag.end(), true);
                    }
                    else
                      if (arg[0] == '-') {
                        cerr << argv[0] << ": ignore unknown option: " << arg << endl;
                      }
                      else {
                        cerr << argv[0] << ": ignore argument: " << arg << endl;
                      }
  }
}

int main(int argc, char*argv[]) {
  InitOption(argc, argv);

  string line;
  int lineno = 0;
  while (getline(cin, line)) {
    ++lineno;
    cout << full2half_A3(line) << endl;
    if (g_Verbose && lineno % 10000 == 0) {
      cerr << lineno << endl;
    }
  }
  return 0;
}
