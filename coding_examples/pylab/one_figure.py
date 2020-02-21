from coding_examples.pylab import *

if __name__ == "__main__":
  data1 = [1, 2, 3, 4]
  pylab.plot(data1, label="data1")
  data2 = [2, 5, 7, 8]
  pylab.plot(data2, label="data2")

  pylab.xlabel("duration in seconds")
  pylab.ylabel("percentage")

  pylab.title("an example")
  pylab.legend()
  pylab.grid()
  pylab.show()
