/**modified date: 3-23-2011
 */

#ifndef LANGUAGE_MODEL_H
#define LANGUAGE_MODEL_H
//#define LM_PRINT

#include "rain_algorithm.h"
#include "lmsri.h"

using namespace rain;

class LanguageModel {
 public:
  LanguageModel(string lm_file, int max_order): lm(NULL), max_order(max_order),
    alpha(log10(exp(1.0))) {
    CountTime ct("loading language model...");
    if ((lm = sriLoadLM(lm_file.c_str(), 1, max_order, 1, 0)) == NULL)
      assert(lm != NULL && !(cerr << "can't open " << lm_file << endl));
  }

  ~LanguageModel() {
    if (lm != NULL) {
      cerr << "unload language model..." << endl;
      sriUnloadLM(lm);
    }
  }

  double get_str_prob(const string& s, const string& ctx = "", int order = -1) {
    VecStr lstr = split(ctx, " "), tmp = split(s, " ");
    int pos = (int)lstr.size();
    lstr.insert(lstr.end(), tmp.begin(), tmp.end());

    double prob = 0;
    string cont;
    order = order <= 0 ? max_order : (order > max_order ? max_order : order);
    for (int i = pos; i < (int)lstr.size(); ++i) {
      cont = join(" ", lstr.begin() + max(0, i - order + 1), lstr.begin() + i);
      prob += get_word_prob(lstr[i], cont);
    }

    return prob;
  }


  double get_word_prob(const string& s, const string& ctx = "") {
    if (s.empty() || (s == "<s>" && ctx.empty()))
      cerr << "warning: in LM.h::get_word_prob(...): s is empty or <s>!" << endl;
    double prob = sriWordProb(lm, s.c_str(), ctx.c_str()) / alpha;
    double r = prob < -100 ? -100 : prob;
#ifdef LM_PRINT
    cout << "word: " << s << ", " << "context: " << ctx << ", " << r << endl;
#endif
    return r;
  }

 private:
  void*               lm;
  int                 max_order;
  double              alpha;
};

#endif

