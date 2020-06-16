package rain;

import java.util.*;

public class WordDict {
  private Map<String, Integer> word2Idx_ = new HashMap<>();

  private List<String> words_ = new ArrayList<>();

  public WordDict() {
  }

  public WordDict(List<String> words) {
    words.stream().forEach(w-> addIdx(w));
  }

  public List<String> getWords() {
    return words_;
  }

  public int addIdx(String word) {
    var idx = word2Idx_.getOrDefault(word, null);
    if (idx != null) {
      return idx;
    }

    words_.add(word);
    word2Idx_.put(word, word2Idx_.size());
    return word2Idx_.size() - 1;
  }

  public int getIdx(String word) {
    return word2Idx_.getOrDefault(word, -1);
  }

  public int getSize() {
    return words_.size();
  }

  public String getWord(int idx) {
    return words_.get(idx);
  }

  public static void main(String[] args) {
    var wd = new WordDict();
    wd.addIdx("summer");
    wd.addIdx("summer1");
    wd.addIdx("summer2");
    wd.addIdx("summer3");

    System.out.println(wd.getSize());
    System.out.println(wd.getWord(3));

    var wd1 = new WordDict(wd.getWords());
    System.out.println(wd1.getWord(0));
  }
}
