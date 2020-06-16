package rain;

public class Counter {
  private int count_ = 0;

  public synchronized int getCount() {
    return count_++;
  }
}