#
# This program serves to exemplify the use of the class FuzzySet for the configuration of the fuzzy set space
#
from fuzzy import FuzzySet as fuzzySet
import matplotlib.pyplot as plt
import numpy as np

#
# Define the list of fuzzy functions names
#
functions = [fuzzySet.FuzzyFunctionsIdentifiers.Sigmoid, fuzzySet.FuzzyFunctionsIdentifiers.Gaussian,
                         fuzzySet.FuzzyFunctionsIdentifiers.Sigmoid]

#
# Define the list of string identifiers for each set
#
identifiers = ["very_low", "low", "medium"]

#
# Define a list with the parameters for each fuzzy function
#
parameters = [[30, -0.5, 0.1], [50, 20, 0.1], [70, 0.5, 0.1]]

#
# The fuzzy set is defined with the FuzzySet class using the list of functions, identifiers and parameters
#
fuzzySetExample = fuzzySet.FuzzySet(functions=functions, identifiers=identifiers, parameters=parameters, end=100, step=1)

#
# Define the range "x" to plot the fuzzy set
#
x = np.arange(0.0, 100, 1)

plt.plot(x, fuzzySetExample.get_fuzzy_set_values())
plt.show()
