#!/usr/bin/env python3
#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

from datetime import date
from examples.py import *

class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

  # a class method to create a Person object by birth year.
  @classmethod
  def fromBirthYear(cls, name, year):
    return cls(name, date.today().year - year)

  # a static method to check if a Person is adult or not.
  @staticmethod
  def isAdult(age):
    return age > 18

person1 = Person('mayank', 21)
person2 = Person.fromBirthYear('mayank', 1996)

print(person1.age)
print(person2.age)

# print the result
print(Person.isAdult(22))

