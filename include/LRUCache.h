#ifndef LRUCACHE_H
#define LRUCACHE_H

#include "rain_algorithm.h"

using namespace rain;

template<class Key, class Value>
class LRUCache{
 public:
  LRUCache(int capacity) {
    capacity_ = capacity;
  }

  bool get(const Key &key, Value &value) {
    auto ite = key2pos_.find(key);
    if (ite == key2pos_.end()) {
      return false;
    }

    value = std::get<1>(*ite->second);
    adjust(ite->second);
    return true;
  }

  void set(const Key &key, const Value &value) {
    auto ite = key2pos_.find(key);
    if (ite != key2pos_.end()) {
      std::get<1>(*ite->second) = value;
      adjust(ite->second);
    }
    else {
      if (SIZE(data_) == capacity_) {
        key2pos_.erase(std::get<0>(data_.front()));
        data_.pop_front(); 
      }

      key2pos_[key] = data_.insert(data_.end(), Record(key, value));
    }
  }

 protected:
  typedef tuple<Key, Value>                 Record;          
  typedef typename list<Record>::iterator   Iterator;

  void adjust(Iterator &ite) {
    if (next(ite) != data_.end()) {
      auto new_iterator = data_.insert(data_.end(), *ite);
      data_.erase(ite);
      ite = new_iterator; 
    }
  }

  int           capacity_ = 0; 
  list<Record>  data_;
  unordered_map<Key, Iterator>  key2pos_;
};
///////////////////////////////////////////////////////////////////////

//int main() {
  //LRUCache<string, double> LMCache(3);
  //LMCache.set("summer", -1.23);
  //LMCache.set("rain", -2.23);
  //LMCache.set("summer rain", -10.23);

  //double lm;
  //if (LMCache.get("summer", lm)) {
    //cout << lm << endl;
  //}
  //LMCache.set("Hello", -0.23);
  //if (LMCache.get("rain", lm)) {
    //cout << lm << endl;
  //}
  //else {
    //cout << "NO" << endl;
  //}

  //return 0;
//}

#endif
