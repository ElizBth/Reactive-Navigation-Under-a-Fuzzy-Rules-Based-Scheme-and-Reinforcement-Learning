from fuzzy import Functions
import numpy as np
import enum


#
# This class contains the names and identifiers of the possible fuzzy functions for this package
#

class FuzzyFunctionsIdentifiers(enum.Enum):
    Triangular = 0
    Trapezoidal = 1
    Gaussian = 2
    Sigmoid = 3
    Bell = 4
    Tangent = 5


#
# This class sets the fuzzy set with fuzzy functions defined by user
#

class FuzzySet:
    _fuzzySpaceIdentifiers = []
    _fuzzySpaceValues = []
    _fuzzyFunctions = []

    #
    # This method initialize the  FuzzySet class with 'n' number of fuzzy functions
    #
    # functions -> list of FuzzyFunctionsIdentifiers class
    # identifiers -> list of strings with set names, defined by expert, example: "high", "red", etc
    # parameters -> function float parameters, i.e. trapezoidal -> (a, b, c, d, alpha), triangular -> (a, b, c, alpha)
    # start -> integer index where function starts
    # end -> integer index where function ends
    # step -> float value which is the step between each evaluation point in functions
    #
    def __init__(self, functions, identifiers, parameters, start=0, end=10, step=0.1):
        self._set_fuzzy_functions(functions, parameters, start, end, step)
        self.__set_fuzzy_set(identifiers, start, end, step)

    #
    # This method is used to set a list with fuzzy functions in the input functions in variable functions
    #
    def _set_fuzzy_functions(self, functions, parameters, start, end, step):
        self._fuzzyFunctions = []
        self._fuzzySpaceIdentifiers = []
        self._fuzzySpaceValues = []
        for index in range(0, len(functions)):
            function = functions[index]
            if function is FuzzyFunctionsIdentifiers.Triangular:
                self._fuzzyFunctions.append(
                    Functions.TriangularFunction(a=parameters[index][0], b=parameters[index][1], c=parameters[index][2],
                                                 alpha=parameters[index][3], start=start, end=end,
                                                 step=step))
            elif function is FuzzyFunctionsIdentifiers.Trapezoidal:
                self._fuzzyFunctions.append(
                    Functions.TrapezoidalFunction(a=parameters[index][0], b=parameters[index][1],
                                                  c=parameters[index][2], d=parameters[index][3],
                                                  alpha=parameters[index][4], start=start, end=end,
                                                  step=step))
            elif function is FuzzyFunctionsIdentifiers.Sigmoid:
                self._fuzzyFunctions.append(
                    Functions.SigmoidFunction(a=parameters[index][0], alpha=parameters[index][1], start=start, end=end,
                                              step=step))
            elif function is FuzzyFunctionsIdentifiers.Gaussian:
                self._fuzzyFunctions.append(
                    Functions.GaussianFunction(c=parameters[index][0], alpha=parameters[index][1], start=start, end=end,
                                               step=step))
            elif function is FuzzyFunctionsIdentifiers.Bell:
                self._fuzzyFunctions.append(
                    Functions.GeneralizedBellCurveFunction(a=parameters[index][0], b=parameters[index][1],
                                                           c=parameters[index][2], alpha=parameters[index][3],
                                                           start=start, end=end, step=step))
            else:
                self._fuzzyFunctions.append(Functions.TangentFunction(start=start, end=end, step=step))

    #
    # This method sets the fuzzy set
    #
    def __set_fuzzy_set(self, identifiers, start, end, step):
        _end_range = int(end / step)
        for index in range(start, _end_range):
            fuzzy_val, fuzzy_val_index = self.__get_fuzzy_functions_max(index)
            self._fuzzySpaceValues.append(fuzzy_val)
            self._fuzzySpaceIdentifiers.append(identifiers[fuzzy_val_index])

    #
    # This method gets the maximum value between fuzzy sets
    #
    def __get_fuzzy_functions_max(self, index):
        max_fuzzy_val = 0
        max_fuzzy_val_index = 0
        for fuzzy_function_index in range(len(self._fuzzyFunctions)):
            if self._fuzzyFunctions[fuzzy_function_index].get_fuzzy_function()[index] > max_fuzzy_val:
                max_fuzzy_val = self._fuzzyFunctions[fuzzy_function_index].get_fuzzy_function()[index]
                max_fuzzy_val_index = fuzzy_function_index
        return max_fuzzy_val, max_fuzzy_val_index

    #
    # This method gets the minimum value between fuzzy sets
    #
    def __get_fuzzy_functions_min(self, index):
        min_fuzzy_val = 1
        min_fuzzy_val_index = 1
        for fuzzy_function_index in range(len(self._fuzzyFunctions)):
            if self._fuzzyFunctions[fuzzy_function_index].get_fuzzy_function()[index] < min_fuzzy_val:
                min_fuzzy_val = self._fuzzyFunctions[fuzzy_function_index].get_fuzzy_function()[index]
                min_fuzzy_val_index = fuzzy_function_index
        return min_fuzzy_val, min_fuzzy_val_index

    #
    # This method gets the product between fuzzy sets
    #
    def __get_fuzzy_functions_product(self, index):
        product_fuzzy_val = 1
        for fuzzy_function_index in range(len(self._fuzzyFunctions)):
            product_fuzzy_val *= self._fuzzyFunctions[fuzzy_function_index].get_fuzzy_function()[index]
        return product_fuzzy_val

    #
    # This method gets the average of the fuzzy function specified with the index
    #
    def get_fuzzy_function_average(self, index):
        return (self._fuzzyFunctions[index]).average

    #
    # This method gets the average of the fuzzy set
    #
    def get_fuzzy_set_average(self):
        return np.average(self._fuzzySpaceValues)

    #
    # This method gets the values of the fuzzy set
    #
    def get_fuzzy_set_values(self):
        return self._fuzzySpaceValues

    #
    # This method gets the identifiers of the fuzzy set
    #
    def get_fuzzy_set_identifiers(self):
        return self._fuzzySpaceIdentifiers

    #
    # This method returns the set identifier with the inserted value
    #
    def get_identifier_from_value(self, value):
        return self._fuzzySpaceIdentifiers[value]

    #
    # This method returns the total fuzzy functions used to generate the fuzzy set
    #
    def get_number_of_functions(self):
        return len(self._fuzzyFunctions)