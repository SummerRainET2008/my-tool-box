#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

from examples.py import *
# import abc

# abc.abstractstaticmethod: deprecated.
# Use 'staticmethod' with 'abstractmethod' instead.

# abc.abstractclassmethod, deprecated.
# Use 'classmethod' with 'abstractmethod' instead.

class Class1(abc.ABC):
  @staticmethod
  @abc.abstractmethod
  def show():
    print(f"This is Class1")

  @abc.abstractmethod
  def hello(self):
    pass

class Class2(Class1):
  @staticmethod
  def show():
    print(f"This is Class2")

  def hello(self):
    print("hello Class2")

if __name__ == "__main__":
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
                     #default=False, help="")
  (options, args) = parser.parse_args()

  obj2 = Class2()
  Class2.show()
  obj2.show()

  obj2.hello()
