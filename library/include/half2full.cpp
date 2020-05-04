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

string half2full_A3(const string& half) {
  size_type i;
  string full;
  bool flag = false;
  for (i = 0; i < half.length(); ++i) {
    char c = half[i];
    if (flag) {
      full += c;
      flag = false;
    }
    else
      if (c < 0) {
        full += c;
        flag = true;
      }
      else
        if (c < ' ') {
          full += c;
        }
        else
          if (c == ' ') {
            if (g_vConvertFlag[c - ' ']) {
              full += '\xA1';
              full += '\xA1';
            }
            else {
              full += c;
            }
          }
          else
            if (c < 127) {
              if (g_vConvertFlag[c - ' ']) {
                full += '\xA3';
                full += (c + 128);
              }
              else {
                full += c;
              }
            }
            else {
              assert(0);
            }
  }
  return full;
}

void Usage(int argc, char* argv[]) {
  cerr << "Usage:" << endl;
  cerr << "    " << argv[0] << " [OPTION]" << endl;
  cerr << "Function:" << endl;
  cerr << "    Convert DBC(half) case to SBC(full) case." << endl;
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
    cout << half2full_A3(line) << endl;
    if (g_Verbose && lineno % 10000 == 0) {
      cerr << lineno << endl;
    }
  }
  return 0;
}
