import math
import numpy as np


class FuzzyFunctions:
    _membershipFunction = []
    _max = 0
    _min = 0
    _average = 0
    _identifier = None

    def __init__(self, a=0, b=0, c=0, d=0, alpha=1, start=0, end=10, step=1, identifier=None):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.alpha = alpha
        self._membershipFunction = []
        self.set_function(start, end, step)
        self.max = np.max(self._membershipFunction)
        self.min = np.min(self._membershipFunction)
        self.average = np.average(self._membershipFunction)
        self.identifier = identifier

    # This method is used to get function values on a point
    # and it is overriding for every fuzzy function
    def get_value_on_point(self, point):
        return point

    # This method is used to build fuzzy functions
    def set_function(self, start, end, step):
        [self._membershipFunction.append(self.get_value_on_point(index)) for index in np.arange(start, end, step)]

    def get_fuzzy_function(self):
        return self._membershipFunction


class TrapezoidalFunction(FuzzyFunctions):

    def get_value_on_point(self, point):
        if self.a <= point <= self.c:
            if self.a == 0 and self.c == 0:
                return 1
            return self.alpha * (point - self.a) / (self.c - self.a)
        elif self.c <= point <= self.d:
            return self.alpha
        elif self.d <= point <= self.b:
            return self.alpha * (point - self.b) / (self.d - self.b)
        else:
            return 0


class TriangularFunction(FuzzyFunctions):

    def get_value_on_point(self, point):
        if self.a <= point <= self.c:
            return self.alpha * (point - self.a) / (self.c - self.a)
        elif self.c <= point <= self.b:
            return self.alpha * (point - self.b) / (self.c - self.b)
        else:
            return 0


class SigmoidFunction(FuzzyFunctions):

    def get_value_on_point(self, point):
        return 1 / (1 + math.exp(-(point - self.a) * self.alpha))


class TangentFunction(FuzzyFunctions):

    def get_value_on_point(self, point):
        return 1 / (1 + math.exp(2 * point))


class GeneralizedBellCurveFunction(FuzzyFunctions):

    def get_value_on_point(self, point):
        return 1 / (1 + abs((point - self.c) / self.a)**(2 * self.b))


class GaussianFunction(FuzzyFunctions):

    def get_value_on_point(self, point):
        return math.exp(-(point - self.c)**2 / (2 * self.alpha**2))
