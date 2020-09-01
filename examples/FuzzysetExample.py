#
# This program serves to exemplify the use of the class FuzzySet for the configuration of the fuzzy set space
#
from fuzzy import FuzzySet as fuzzySet
import matplotlib.pyplot as plt
import numpy as np

#
# Define the list of fuzzy functions names
#
functions = [fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal, fuzzySet.FuzzyFunctionsIdentifiers.Triangular,
                         fuzzySet.FuzzyFunctionsIdentifiers.Triangular, fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal]

#
# Define the list of string identifiers for each set
#
identifiers = ["very_low", "low", "medium", "high"]

#
# Define a list with the parameters for each fuzzy function
#
parameters = [[0, 40, 0, 20, 1], [20, 60, 40, 1], [40, 80, 60, 1], [60, 100, 80, 100, 1]]

#
# The fuzzy set is defined with the FuzzySet class using the list of functions, identifiers and parameters
#
fuzzySetExample = fuzzySet.FuzzySet(functions=functions, identifiers=identifiers, parameters=parameters, end=100, step=1)

#
# Define the range "x" to plot the fuzzy set
#
#x = np.arange(0.0, 100, 1)

#plt.plot(x, fuzzySetExample.get_fuzzy_set_values())
#plt.show()
