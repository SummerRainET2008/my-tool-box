#ifndef RAIN_ALGORITHM
#define RAIN_ALGORITHM

/////////////////// Examples ////////////////////
// Standard definition of operator Compare class.
// bool operator ()(const Type &a, const Type &b) const {
//      // The second "<" swaps "a" and "b".
//      return a.f1 < b.f1 || !(b.f1 < a.f1) && a.f2 < b.f2;
// }
//
// cout << boost::format("writing %1%,  x=%2% : %3%-th try") \
//        % "toto" % 40.23 % 50 << endl;
// boost::trim(name);
// boost::replace_all(name, "summer", "rain");
//
//
// valarray<double> v = {1, 2, 3, 4}, result;
// result = exp(v);                         // It's wrong.
// result = valarray<double>(exp(v));       // It's right.
//          OR
// valarray<double> v = {1, 2, 3, 4}, result(3);
// result = exp(v);                         // It's right;
//
// cout << typeid(10).name() << endl;
//
//
// double f =9876543210.012345678901234567890123456789;
// std::cout << std::setprecision(5) << f << '\n';
// std::cout << std::setprecision(9) << f << '\n';
// std::cout << std::fixed;
// std::cout << std::setprecision(5) << f << '\n';
// std::cout << std::setprecision(9) << f << '\n';

/*Output:
* 9.8765e+09
* 9.87654321e+09
* 9876543210.01235
* 9876543210.012346268
*/

// cout.precision(6);
// cout.unsetf(ios::floatfield);

/////////////////////////////////////////////////

///////////////// C headers //////////////////
// No need to add  ahead of all functions in these header files.
#include <ctime>
#include <cassert>
#include <cmath>
//#include <cctype>

///////////////// C++ headers ////////////////
// container header files
//#include <bitset>
//#include <deque>
//#include <forward_list>  
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <unordered_map>
//#include <unordered_set>
#include <vector>

// input-output header files
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

// utility header files
#include <algorithm>
//#include <complex>
//#include <exception>
//#include <functional>
//#include <initializer_list>
//#include <iterator>
#include <climits>
//#include <limits>
//#include <locale>
//#include <memory>
//#include <mutex>
//#include <numeric>
#include <random>
//#include <ratio>
//#include <regex>
//#include <stdexcept>
#include <string>
//#include <tuple>
#include <thread>
//#include <type_traits>
//#include <utility>
// all functions on valarray do NOT return valarray actually!
// Must add valarray<Type>(function(exp(..)));
#include <valarray>

using namespace std;
using namespace std::placeholders;    // adds visibility of _1, _2, _3,...

///////////////// other platforms ////////////////
// getpid()
//#include <unistd.h>

#include <boost/program_options.hpp>
#include <boost/format.hpp>
//#include <boost/algorithm/string.hpp>

namespace po = boost::program_options;
using boost::format;

//////////////////////////////////////////////////

#ifdef MY_DEBUG
  #define GET(data, idx) data.at(idx)
#else
  #define GET(data, idx) data[idx]
#endif

#define SIZE(container) int((container).size())

#define OUTPUT(begin, end, Type, stream) \
  copy(begin, end, ostream_iterator<Type>(stream, " "));

#define PRINT_INF(x) \
  cerr << __FILE__ << ":" << __LINE__ << ":" << (x) << endl;

#define OPEN_RFILE(fin, name) \
  ifstream fin(name); \
  assert(fin.good() || !(cerr << "can't open " << (name) << endl));

#define OPEN_WFILE(fou, name) \
  ofstream fou(name); \
  assert(fou.good() || !(cerr << "can't open " << (name) << endl));

#define EPSILON             1e-6
#define INF                 1e100
#define DOUBLE_MAX          numeric_limits<double>::max()
#define DOUBLE_MIN          -numeric_limits<double>::max()
#define VVec(type)          vector<vector<type>>

typedef long long           int64;
typedef unsigned long long  uint64;
typedef vector<int>         VecInteger; 
typedef vector<unsigned>    VecUInteger; 
typedef vector<double>      VecDouble;  
typedef vector<string>      VecString;

typedef vector<int>         VecInt; 
typedef vector<unsigned>    VecUInt; 
typedef vector<double>      VecDbe;  
typedef vector<string>      VecStr;

namespace rain {
class DisjointSet {
 public:
  DisjointSet(int size) {
    fathers_.resize(size, -1);
    sizes_.resize(size, 1);
    cluster_size_ = size;
  }

  // Adding the judement that which cluster is greater in size, would
  // prevent from some extreme cases. 
  // For example, combine(0, 1), combine(1, 2), ..., combine(m, m + 1), ...
  // This would lead to maximum recursion limit exceeded.
  void combine(int a, int b) {
    int f1 = getFather(a), f2 = getFather(b);
    if (f1 != f2) {
      if (sizes_[f1] >= sizes_[f2]) {
        fathers_[f2] = f1;
        sizes_[f1] += sizes_[f2];
      }
      else {
        fathers_[f1] = f2;
        sizes_[f2] += sizes_[f1];
      }
      cluster_size_ -= 1;
    }
  }

  int size() const {
    return cluster_size_;
  }

  int getFather(int a) {
    if (fathers_[a] == -1) {
      return a;
    }
    fathers_[a] = getFather(fathers_[a]);
    return fathers_[a];
  }

 protected:
  vector<int> fathers_, sizes_;
  int cluster_size_;
};

class RandInt {
 public:
  // We should do srand(seed) at first beginning of our program.

  // [begin, end)
  int operator ()(int begin, int end) {
    assert(begin < end);
    auto rnd = rand() ^ (unsigned(rand()) << 15) ^ (unsigned(rand()) << 30);
    return rnd % (end - begin) + begin;
  }
};

class IndexManager {
 public:
  int add(const string &label) {
    auto ite = label2index_.find(label);
    if (ite == label2index_.end()) {
      label2index_[label] = SIZE(labels_);
      labels_.push_back(label);
      return SIZE(labels_) - 1;
    }
    else {
      return ite->second;
    }
  }

  int getIndex(const string &label) const {
    auto ite = label2index_.find(label);
    assert(ite != label2index_.end());
    return ite->second;
  }

  string getName(int index) const {
    return labels_.at(index);
  }

  int size() const {
    return SIZE(labels_);
  }

  void clear() {
    labels_.clear();
    label2index_.clear();
  }

 protected:
  vector<string>              labels_;
  unordered_map<string, int>  label2index_;
};

template<class Type>
ostream& operator << (ostream &stream, const vector<Type> &vec) {
  for (const auto &d: vec) {
    stream << d << " ";
  }
  return stream;
}

template<class Type>
void reservoirSample(const vector<Type> &data, vector<Type> &out, int k) {
  out.clear();
  if (k < 1) {
    return;
  }

  for (int p = 0; p < k; ++p) {
    out.push_back(data[p]);
  }

  //auto myrand = RandInt(0);
  auto myrand = RandInt();
  for (int p = k; p < SIZE(data); ++p) {
    auto pos = myrand(0, p);
    //cout << "p: " << p << ", " << pos << endl;
    if (pos < k) {
      out[pos] = data[p];
    }
  }
}

// logSum(vector) or logSum<initializer_list<double>>({1, 2, 3, 4});
template<class Type>
double log_sum(const Type &data) {
  double maxv = *max_element(begin(data), end(data));
  double ret = 0;
  for (auto e: data) {
    ret += exp(e - maxv);
  }
  return log(ret) + maxv;
}

template<class Vector>
double BLEU(const Vector &ngram, bool sentlev = false) {
  //  ngram[10] = {1matched/1-num ..4matched/4-num ref-len valid-hyp-num}
  //  1. If to compute sbp_BLEU, valid-hyp-num is the min(ref-len, hyp-len),
  //  2. When computing BLEU on sentence level, it's better to throw
  //  penalty factor in practice.
  assert(ngram.size() == 10);
  double precs = 1;
  double ratio = (sentlev ? 
                  1 : exp(min(0.0, 1 - double(ngram[8]) / ngram[9])));
  double delta = sentlev ? 1 : 0;
  for (int ni = 0; ni < 4; ++ni) {
    //precs *= ngram[ni * 2] / (ngram[ni * 2 + 1] + 1e-6);
    precs *= (ngram[ni * 2] + delta) / (ngram[ni * 2 + 1] + delta);
  }
  precs = pow(precs, 0.25);
  return precs * ratio;
}

template<class ConstPointer>
string join(string s, ConstPointer begin, ConstPointer end) {
  if (begin == end) {
    return "";
  }
  else {
    string ret = *begin++;
    while (begin != end) {
      ret += s + *begin++;
    }
    return ret;
  }
}

// eg. split("   a   b", " ") will return [a, b].
inline vector<string> split(const string &src, string delimit) {
  vector<string> ret;
  size_t np = 0, p = 0, r;
  while ((r = src.find(delimit, p)) != string::npos) {
    if (r > np) {
      ret.push_back(src.substr(np, r - np));
    }
    np = r + delimit.size();
    p = r + 1;
  }
  if (np < src.size()) {
    ret.push_back(src.substr(np));
  }
  return ret;
}

inline vector<string> split(const string &src) {
  istringstream iss(src);
  vector<string> ret;
  string w;
  while (iss >> w) {
    ret.push_back(w);
  }
  return ret;
}

// double f1 = 1.234567891234567891234567891234567e30;
// double f2 = 1.234567891234566891234567891234567e30;
//
// cout << setprecision(100) << boolalpha;
// cout << "f1: " << f1 << "\n";
// cout << "f2: " << f2 << "\n";
// cout << "relevant difference: " << eq(f1, f2) << endl;
// cout << endl;
// cout << "absolute difference: " << eq1(f1, f2) << endl;
// cout << "direct   comparison: " << (f1 < f2) << endl;

// The precision of double is 17;   1.234567891234567e20;
// The precision of float  is 9;    1.23456789e20;
// No need to define strictGreater and strictLess;
// In "The art of computer programming by Knuth", this is also called \
// approximatelyEqual;
// bool approximatelyEqual(float a, float b, float epsilon) {
//     return fabs(a - b) <= ( (fabs(a) < fabs(b) ? \
//     fabs(b) : fabs(a)) * epsilon);
// }

// bool essentiallyEqual(float a, float b, float epsilon) {
// return fabs(a - b) <= ( (fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * epsilon);
// }

// bool definitelyGreaterThan(float a, float b, float epsilon) {
// return (a - b) > ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
// }

// bool definitelyLessThan(float a, float b, float epsilon) {
// return (b - a) > ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
// }

//inline bool eq(double m, double n, double prec = 1e-13) {
//return fabs(m - n) <= max(fabs(m), fabs(n)) * prec;
//}

// When judge whether is zero, can not use eq(a, EPSILON).
// Should be eq(a, 0);
inline bool eq(double a, double b) {
  return fabs(a - b) <= EPSILON;
}

inline bool strictLess(double a, double b) {
  return !eq(a, b) && a < b;
}

inline bool strictGreater(double a, double b) {
  return !eq(a, b) && a > b;
}

inline bool lessAndEq(double a, double b) {
  return !strictGreater(a, b);
}

inline bool greaterAndEq(double a, double b) {
  return !strictLess(a, b);
}

inline bool startsWith(const string &str, string prefix) {
  return (prefix.size() <= str.size() && 
          str.compare(0, prefix.size(), prefix) == 0);
}

inline bool endsWith(const string &str, string postfix) {
  return (postfix.size() <= str.size() && 
          str.compare(str.size() - postfix.size(), 
                      postfix.size(), postfix) == 0);
}

// return memory used, MB
inline double get_memory() {
#ifdef __APPLE__
  cerr << "Cannot get memory in Mac" << endl;
  return -3.1415926;
#else
  string fname = string("/proc/") + to_string(getpid()) + "/status";
  ifstream fin(fname);
  if (!fin.good()) {
    cerr << "Failed to get memory" << endl;
    return -3.1415926;
  }
  else {
    string line;
    while (getline(fin, line) && !startsWith(line, "VmSize")) {
    }
    return stod(split(line).at(1)) / 1000;
  }
#endif
}

class CountTime {
 public:
  CountTime(string inf, bool alive = true): 
      alive_(alive), inf_(inf), start_(clock()) {
        // alive: when False, CountTime forbids any output;
        if (alive_) {
          cout << string(10, '*') + string(50, '-') << endl;
          cout << format("(start - {%1%}") % inf_ << endl;
        }
      }

  double getHours() const {
    return getSeconds() / 3600;
  }

  double getMinutes() const {
    return getSeconds() / 60;
  }

  double getSeconds() const {
    auto finish = clock();
    double tsecond = (finish - start_) / (double)CLOCKS_PER_SEC;
    return tsecond;
  }

  ~CountTime() {
    if (alive_) {
      double tsecond = getSeconds();
      int h = (int)(tsecond / (60 * 60));
      int m = (int)((tsecond - h * 60 * 60) / 60);
      double s = tsecond - h * 60 * 60 - m * 60;

      cout << (format("(finish - %1%): %2% h %3% m %4% s") 
               %inf_ %h %m %s) << endl;
      cout << string(50, '-') + string(10, '*') << endl;
    }
  }

 private:
  bool                alive_;
  string              inf_;
  decltype(clock())   start_;
};

inline void l1Norm(VecDouble &weight, double norm = 1) {
  double wsum = 0;
  for (auto w: weight) {
    wsum += fabs(w);    
  }

  if (eq(wsum, 0)) {
    cout << "Warning in l1Norm: 0 vector found" << endl;
    return;
  }

  double ratio = norm / wsum; 
  for (auto &w: weight) {
    w *= ratio;
  }
}
}

#endif

