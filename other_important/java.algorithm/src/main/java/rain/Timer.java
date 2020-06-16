package rain;

public class Timer {
  private long start_ = System.currentTimeMillis();
  private String message_;

  public Timer(String message) {
    message_ = message;
  }

  public void reset() {
    start_ = System.currentTimeMillis();
  }

  public double getSeconds() {
    return (System.currentTimeMillis() - start_) / 1000.;
  }

  public double getMinutes() {
    return getSeconds() / 60;
  }

  public void stop() {
    System.out.printf("'%s' takes %.3f seconds\n", message_, getSeconds());
  }
};


