package rain;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import org.apache.commons.lang3.ArrayUtils;
import static java.lang.System.out;
import static java.util.stream.Collectors.counting;
import static java.util.stream.Collectors.groupingBy;

public class Alg {
  public static double EPSILON = 1e-8;
  public static Random rand_ = new Random();

  public static void setParallelism(int theadNum) {
    var key = "java.util.concurrent.ForkJoinPool.common.parallelism";
    System.setProperty(key, Integer.toString(theadNum));
  }

  public static int executeCmd(String cmd) {
    // quite simple comamnds
    try {
      Runtime run = Runtime.getRuntime();
      Process pr = run.exec(cmd);
      var ret = pr.waitFor();
      return ret;
      /*BufferedReader buf = new BufferedReader(new InputStreamReader(pr.getInputStream()));

      String line;
      while ((line = buf.readLine()) != null) {
        System.out.println(line);
      }*/
    }
    catch (Exception error) {
      return 1;
    }
  }

  public static double logSum(List nums) {
    var maxV = (Double)Collections.max(nums);
    var resSum = 0d;
    for (var num: nums) {
      resSum += Math.exp((Double)num - maxV);
    }

    return maxV + Math.log(resSum);
  }

  public static double logSum(double[] nums) {
    var maxV = nums[argMax(nums)];
    var resSum = 0d;
    for (var num: nums) {
      resSum += Math.exp(num - maxV);
    }

    return maxV + Math.log(resSum);
  }

  public static float logSum(float[] nums) {
    var maxV = nums[argMax(nums)];
    var resSum = 0d;
    for (var num: nums) {
      resSum += Math.exp(num - maxV);
    }

    return maxV + (float)Math.log(resSum);
  }

  public static int argMax(List nums) {
    var maxV = (double)nums.get(0);
    var ret = 0;
    for (var p = 1; p < nums.size(); ++p) {
      var v = (double)nums.get(p);
      if (v > maxV) {
        maxV = v;
        ret = p;
      }
    }
    return ret;
  }

  public static int argMax(double[] nums) {
    var maxV = nums[0];
    var ret = 0;
    for (var p = 1; p < nums.length; ++p) {
      if (nums[p] > maxV) {
        maxV = nums[p];
        ret = p;
      }
    }
    return ret;
  }

  public static int argMax(float[] nums) {
    var maxV = nums[0];
    var ret = 0;
    for (var p = 1; p < nums.length; ++p) {
      if (nums[p] > maxV) {
        maxV = nums[p];
        ret = p;
      }
    }
    return ret;
  }

  public static int argMax(int[] nums) {
    var maxV = nums[0];
    var ret = 0;
    for (var p = 1; p < nums.length; ++p) {
      if (nums[p] > maxV) {
        maxV = nums[p];
        ret = p;
      }
    }
    return ret;
  }

  public static int argMin(List nums) {
    var maxV = (double)nums.get(0);
    var ret = 0;
    for (var p = 1; p < nums.size(); ++p) {
      var v = (double)nums.get(p);
      if (v < maxV) {
        maxV = v;
        ret = p;
      }
    }
    return ret;
  }

  public static int argMin(double[] nums) {
    var maxV = nums[0];
    var ret = 0;
    for (var p = 1; p < nums.length; ++p) {
      if (nums[p] < maxV) {
        maxV = nums[p];
        ret = p;
      }
    }
    return ret;
  }

  public static int argMin(float[] nums) {
    var maxV = nums[0];
    var ret = 0;
    for (var p = 1; p < nums.length; ++p) {
      if (nums[p] < maxV) {
        maxV = nums[p];
        ret = p;
      }
    }
    return ret;
  }

   public static int argMin(int[] nums) {
    var maxV = nums[0];
    var ret = 0;
    for (var p = 1; p < nums.length; ++p) {
      if (nums[p] < maxV) {
        maxV = nums[p];
        ret = p;
      }
    }
    return ret;
  }

  public static <Type> Type last(List<Type> data) {
    return data.get(data.size() - 1);
  }

  // We do not put back, so num <= data.size();
  public static <Type> List<Type> sample(List<Type> data, int num) {
    if (num <= 0) {
      System.out.printf("Warning: sample size=%d <= 0\n", num);
      return new ArrayList<>();
    }
    if (num >= data.size()) {
      return data.subList(0, data.size());
    }

    for (var p = 0; p < num; ++p) {
      var pos = rand_.nextInt(data.size() - p);
      Collections.swap(data, p, p + pos);
    }
    return data.subList(0, num);
  }

  public static Writer openWriteFile(String fname) throws IOException {
    return new BufferedWriter(new FileWriter(fname));
  }

  public static Stream<String> openReadFileStream(String fname)
    throws IOException {
    return Files.lines(Paths.get(fname)).map(String::trim);
  }

  public static List<String> openReadFileLines(String fname)
    throws IOException {
    return openReadFileStream(fname).collect(Collectors.toList());
  }

  public static BufferedReader openReadFile(String fname) throws IOException {
    return new BufferedReader(new FileReader(fname));
  }

  public static <Type> Map<Type, Long> countWords(Stream<Type> strStream) {
    return strStream.collect(groupingBy(key-> key, counting()));
  }

  public static Map<String, String> extractAttribute(String[] blocks) {
    Map<String, String> ret = new TreeMap<>();
    for (var block: blocks) {
      int p = block.indexOf("=");
      if (p == -1) {
        out.println("found wrong block: " + block);
        return new TreeMap<>();
      }

      ret.put(block.substring(0, p), block.substring(p + 1));
    }
    return ret;
  }

  public static boolean eq(double a, double b) {
    return Math.abs(a - b) <= EPSILON;
  }

  public static boolean strictLess(double a, double b) {
    return !eq(a, b) && a < b;
  }

  public static boolean strictGreater(double a, double b) {
    return !eq(a, b) && a > b;
  }

  public static <Type> void resize(List<Type> list, int size, Type value) {
    while (list.size() < size) {
      list.add(value);
    }
    while (list.size() > size) {
      list.remove(list.size() - 1);
    }
  }

  public static double[] l1Norm(double[] weight, double norm) {
    double wsum = 0.;
    for (var w: weight) {
      wsum += Math.abs(w);
    }

    if (eq(wsum, 0)) {
      out.println("Warning in l1Norm: 0 vector found");
      return weight;
    }

    double ratio = norm / wsum;
    var ret = new double[weight.length];
    for (int p = 0; p < weight.length; ++p) {
      ret[p] = weight[p] * ratio;
    }
    return ret;
  }

  //public static <Type> void update(List<Type> data, int pos, Type value) {
  //data.set(pos, data.get(pos) + value);
  //}

  public static void main(String[] argv) {
    var list0 = List.of(1d, 2d, 3d);
    System.out.println(logSum(list0));

    var timer = new Timer("Test Alg");
    out.println("Hello rain.Alg");

    var list = Arrays.asList(1, 2, 3, 4, 5);
    out.println(Alg.last(list));

    var data = new ArrayList<>();
    Alg.resize(data, 10, 0.);
    out.println(data.size());

    double[] weights = {
      1, 2, 3, 4,
      };
    var weightsL1 = Alg.l1Norm(weights, 1.);
    //    out.println(Alg.last(weights));

    out.println(Arrays.asList(ArrayUtils.toObject(weightsL1)));

    out.println(Alg.extractAttribute("name=\tage=30   ".split("\t")));
    out.println(Alg.extractAttribute("n=n=summer\tage=30   ".split("\t")));

    var stream = "a b c d a b d g e 3 4 1 1 1 2 2 3";
    System.out.println(countWords(Arrays.stream(stream.split("\\s+"))));

    System.out.println("seconds: " + timer.getSeconds());
    System.out.println("minutes: " + timer.getMinutes());
    timer.reset();

    timer.stop();

    var data1 = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8);
    System.out.println(sample(data1, 3));
    System.out.println(sample(data1, -1));
    System.out.println(sample(data1, 10));
  }
};
