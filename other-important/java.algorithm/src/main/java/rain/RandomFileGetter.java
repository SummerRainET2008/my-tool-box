package rain;

import java.io.*;
import java.util.*;

public class RandomFileGetter {
  private BufferedReader reader_;
  private List<String> buffer_ = new ArrayList<>();
  private boolean isClosed_ = true;
  private Random random_;
  private int bufferSize_;

  public RandomFileGetter(String fname) {
    this(fname, 1, new Random());
  }

  public RandomFileGetter(String fname, int bufferSize) {
    this(fname, bufferSize, new Random());
  }

  public RandomFileGetter(String fname, int bufferSize, Random random) {
    bufferSize_ = bufferSize;
    random_ = random;

    try {
      reader_ = new BufferedReader(new FileReader(fname));
      isClosed_ = false;
    }
    catch (FileNotFoundException error) {
      System.out.println(error.getMessage());
    }
  }

  public synchronized String readLine() {
    try {
      while (!isClosed_ && buffer_.size() < bufferSize_) {
        var line = reader_.readLine();
        if (line == null) {
          isClosed_ = true;
          reader_.close();
          break;
        }
        buffer_.add(line);
      }

      if (buffer_.isEmpty()) {
        return null;
      }
      int index = random_.nextInt(buffer_.size());
      int lastIndex = buffer_.size() - 1;
      var ret = buffer_.get(index);
      buffer_.set(index, buffer_.get(lastIndex));
      buffer_.remove(lastIndex);
      return ret;
    }
    catch (IOException error) {
      System.out.println(error.getMessage());
      isClosed_ = true;
      return readLine();
    }
  }
}
