from examples.pylab import *

if __name__ == "__main__":
  pylab.figure(0)
  pylab.plot([1,2,3])

  pylab.figure(1)
  pylab.plot([10, 20, 30])

  pylab.figure(0)
  pylab.plot([4, 5, 6])

  pylab.figure(1)
  pylab.plot([40, 50, 60])

  pylab.show()