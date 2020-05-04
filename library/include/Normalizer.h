#ifndef NORMALIZER_H
#define NORMALIZER_H

/*
   modified on 2009/1/5
   */

#include "mitel_algorithm.h"

using namespace mitel;

class Normalizer {
 public:
  string norm(const string &hyp, bool upper = false, bool headline = false) {
    string nhyp = strip(hyp);
    if (nhyp.empty())
      return nhyp;
    else {
      if (upper)
        nhyp[0] = toupper(nhyp[0]);
      capitalize_subsentence(nhyp);
      dropOOV(nhyp);
      if (headline)
        deal_headline(nhyp);
      punc(nhyp);
      return nhyp;
    }
  }
 private:
  void capitalize_subsentence(string& str) {
    istringstream buf(str);
    vector<string> wordVec;
    string w;

    while (buf >> w)
      wordVec.push_back(w);

    for (int i = 1; i < (int)wordVec.size(); i++) {
      if (wordVec[i - 1] == "." || wordVec[i - 1] == "?" || wordVec[i - 1] == "!")
        wordVec[i][0] = toupper(wordVec[i][0]);

      if (i >= 2 && wordVec[i - 1] == "\"" && (wordVec[i - 2] == ":"
          || wordVec[i - 2] == ","))
        wordVec[i][0] = toupper(wordVec[i][0]);
    }

    str = join(" ", wordVec);
  }

  void dropOOV(string& str) {
    istringstream buf(str);
    string w;
    string str1;

    while (buf >> w) {
      const char *buffer = w.c_str();

      if ((unsigned char)(*buffer) >= 176 && (unsigned char)(*buffer) <= 247)
        continue;
      str1 += w + " ";
    }

    str = str1;
  }

  void deal_headline(string& str) {
    map<string, int> m;
    m["a"] = 1;
    m["an"] = 1;
    m["as"] = 1;
    m["and"] = 1;
    m["at"] = 1;
    m["by"] = 1;
    m["for"] = 1;
    m["from"] = 1;
    m["in"] = 1;
    m["of"] = 1;
    m["on"] = 1;
    m["or"] = 1;
    m["that"] = 1;
    m["the"] = 1;
    m["this"] = 1;
    m["to"] = 1;
    m["with"] = 1;
    m["within"] = 1;
    m["without"] = 1;
    m["after"] = 1;
    m["into"] = 1;
    m["out"] = 1;
    m["before"] = 1;
    m["among"] = 1;
    m["under"] = 1;
    m["off"] = 1;
    m["up"] = 1;

    vector<string> wordVec;
    istringstream buf(str);
    string w;

    while (buf >> w) {
      if (islower((int)w[0])) {
        map<string, int>::iterator iter = m.find(w);
        if (iter == m.end())
          w[0] = toupper((int)w[0]);
      }

      wordVec.push_back(w);
    }

    str = join(" ", wordVec);
  }

  void punc(string& str) {
    replace(str, " 's", "'s");
    replace(str, "s '", "s'");
    replace(str, " n't", "n't");
    replace(str, ", ,", ",");
    replace(str, ", .", ".");
    replace(str, ". ,", ",");
  }
};

#endif
