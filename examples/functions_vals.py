from fuzzy import Functions as fuzzyFunctions
import matplotlib.pyplot as plt
import numpy as np
from fuzzy import FuzzySet as fuzzySet

## Version 1
battery_functions = [fuzzySet.FuzzyFunctionsIdentifiers.Sigmoid, fuzzySet.FuzzyFunctionsIdentifiers.Gaussian,
                     fuzzySet.FuzzyFunctionsIdentifiers.Sigmoid]
battery_identifiers = ["low", "medium", "high"]
battery_parameters = [[30, -0.5, 0.1], [50, 20, 0.1], [70, 0.5, 0.1]]
target_functions = [fuzzySet.FuzzyFunctionsIdentifiers.Sigmoid, fuzzySet.FuzzyFunctionsIdentifiers.Gaussian,
                    fuzzySet.FuzzyFunctionsIdentifiers.Sigmoid]
target_identifiers = ["close", "near", "far"]
target_parameters = [[30, -0.5, 0.1], [50, 20, 0.1], [70, 0.5, 0.1]]
station_functions = [fuzzySet.FuzzyFunctionsIdentifiers.Sigmoid, fuzzySet.FuzzyFunctionsIdentifiers.Gaussian,
                     fuzzySet.FuzzyFunctionsIdentifiers.Sigmoid]
station_identifiers = ["close", "near", "far"]
station_parameters = [[30, -0.5, 0.1], [50, 20, 0.1], [70, 0.5, 0.1]]

battery_fuzzy_set = fuzzySet.FuzzySet(functions=battery_functions, identifiers=battery_identifiers,
                                      parameters=battery_parameters, step=1, end=101)

target_fuzzy_set = fuzzySet.FuzzySet(functions=target_functions, identifiers=target_identifiers,
                                     parameters=target_parameters, step=1, end=101)

station_fuzzy_set = fuzzySet.FuzzySet(functions=station_functions, identifiers=station_identifiers,
                                      parameters=station_parameters, step=1, end=101)

fuzzy_set_list = [battery_fuzzy_set.get_fuzzy_functions()[0].get_fuzzy_function()[0:], battery_fuzzy_set.get_fuzzy_functions()[1].get_fuzzy_function()[0:], battery_fuzzy_set.get_fuzzy_functions()[2].get_fuzzy_function()[0:],
                  target_fuzzy_set.get_fuzzy_functions()[0].get_fuzzy_function()[0:], target_fuzzy_set.get_fuzzy_functions()[1].get_fuzzy_function()[0:], target_fuzzy_set.get_fuzzy_functions()[2].get_fuzzy_function()[0:],
                  station_fuzzy_set.get_fuzzy_functions()[0].get_fuzzy_function()[0:], station_fuzzy_set.get_fuzzy_functions()[1].get_fuzzy_function()[0:], station_fuzzy_set.get_fuzzy_functions()[2].get_fuzzy_function()[0:]]

np.savetxt("sets_v1.csv",
                       np.transpose(fuzzy_set_list),
                       delimiter=", ",
                       fmt='% s')

## Version 2

battery_functions = [fuzzySet.FuzzyFunctionsIdentifiers.Sigmoid, fuzzySet.FuzzyFunctionsIdentifiers.Triangular,
                     fuzzySet.FuzzyFunctionsIdentifiers.Sigmoid]
battery_identifiers = ["low", "medium", "high"]
battery_parameters = [[30, -0.5, 0.1], [0, 100, 50, 0.1], [70, 0.5, 0.1]]
target_functions = [fuzzySet.FuzzyFunctionsIdentifiers.Triangular, fuzzySet.FuzzyFunctionsIdentifiers.Gaussian,
                    fuzzySet.FuzzyFunctionsIdentifiers.Triangular]
target_identifiers = ["close", "near", "far"]
target_parameters = [[-1, 50, 0, 0.1], [50, 20, 0.1], [50, 101, 100, 0.1]]
station_functions = [fuzzySet.FuzzyFunctionsIdentifiers.Triangular, fuzzySet.FuzzyFunctionsIdentifiers.Gaussian,
                     fuzzySet.FuzzyFunctionsIdentifiers.Triangular]
station_identifiers = ["close", "near", "far"]
station_parameters = [[-1, 50, 0, 0.1], [50, 20, 0.1], [50, 101, 100, 0.1]]


battery_fuzzy_set = fuzzySet.FuzzySet(functions=battery_functions, identifiers=battery_identifiers,
                                      parameters=battery_parameters, step=1, end=101)

target_fuzzy_set = fuzzySet.FuzzySet(functions=target_functions, identifiers=target_identifiers,
                                     parameters=target_parameters, step=1, end=101)

station_fuzzy_set = fuzzySet.FuzzySet(functions=station_functions, identifiers=station_identifiers,
                                      parameters=station_parameters, step=1, end=101)

fuzzy_set_list = [battery_fuzzy_set.get_fuzzy_functions()[0].get_fuzzy_function()[0:], battery_fuzzy_set.get_fuzzy_functions()[1].get_fuzzy_function()[0:], battery_fuzzy_set.get_fuzzy_functions()[2].get_fuzzy_function()[0:],
                  target_fuzzy_set.get_fuzzy_functions()[0].get_fuzzy_function()[0:], target_fuzzy_set.get_fuzzy_functions()[1].get_fuzzy_function()[0:], target_fuzzy_set.get_fuzzy_functions()[2].get_fuzzy_function()[0:],
                  station_fuzzy_set.get_fuzzy_functions()[0].get_fuzzy_function()[0:], station_fuzzy_set.get_fuzzy_functions()[1].get_fuzzy_function()[0:], station_fuzzy_set.get_fuzzy_functions()[2].get_fuzzy_function()[0:]]

np.savetxt("sets_v2.csv",
                       np.transpose(fuzzy_set_list),
                       delimiter=", ",
                       fmt='% s')

## Version 3

battery_functions = [fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal, fuzzySet.FuzzyFunctionsIdentifiers.Triangular,
                     fuzzySet.FuzzyFunctionsIdentifiers.Triangular, fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal]
battery_identifiers = ["very_low", "low", "medium", "high"]
battery_parameters = [[0, 40, 0, 20, 1], [20, 60, 40, 1], [40, 80, 60, 1], [60, 100, 80, 100, 1]]

target_functions = [fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal, fuzzySet.FuzzyFunctionsIdentifiers.Triangular,
                    fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal]
target_identifiers = ["close", "near", "far"]
target_parameters = [[0, 50, 0, 20, 1], [20, 80, 50, 1], [50, 100, 80, 100, 1]]

station_functions = [fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal, fuzzySet.FuzzyFunctionsIdentifiers.Triangular,
                     fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal]
station_identifiers = ["close", "near", "far"]
station_parameters = [[0, 50, 0, 20, 1], [20, 80, 50, 1], [50, 100, 80, 100, 1]]

battery_fuzzy_set = fuzzySet.FuzzySet(functions=battery_functions, identifiers=battery_identifiers,
                                      parameters=battery_parameters, step=1, end=101)

target_fuzzy_set = fuzzySet.FuzzySet(functions=target_functions, identifiers=target_identifiers,
                                     parameters=target_parameters, step=1, end=101)

station_fuzzy_set = fuzzySet.FuzzySet(functions=station_functions, identifiers=station_identifiers,
                                      parameters=station_parameters, step=1, end=101)

fuzzy_set_list = [battery_fuzzy_set.get_fuzzy_functions()[0].get_fuzzy_function()[0:], battery_fuzzy_set.get_fuzzy_functions()[1].get_fuzzy_function()[0:], battery_fuzzy_set.get_fuzzy_functions()[2].get_fuzzy_function()[0:],  battery_fuzzy_set.get_fuzzy_functions()[3].get_fuzzy_function()[0:],
                  target_fuzzy_set.get_fuzzy_functions()[0].get_fuzzy_function()[0:], target_fuzzy_set.get_fuzzy_functions()[1].get_fuzzy_function()[0:], target_fuzzy_set.get_fuzzy_functions()[2].get_fuzzy_function()[0:],
                  station_fuzzy_set.get_fuzzy_functions()[0].get_fuzzy_function()[0:], station_fuzzy_set.get_fuzzy_functions()[1].get_fuzzy_function()[0:], station_fuzzy_set.get_fuzzy_functions()[2].get_fuzzy_function()[0:]]

np.savetxt("sets_v3.csv",
                       np.transpose(fuzzy_set_list),
                       delimiter=", ",
                       fmt='% s')


battery_functions = [fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal, fuzzySet.FuzzyFunctionsIdentifiers.Triangular,
                     fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal]
battery_identifiers = ["low", "medium", "high"]
battery_parameters = [[0, 50, 0, 20, 1], [20, 80, 50, 1], [50, 100, 80, 100, 1]]

target_functions = [fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal, fuzzySet.FuzzyFunctionsIdentifiers.Triangular,
                    fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal]
target_identifiers = ["close", "near", "far"]
target_parameters = [[0, 50, 0, 20, 1], [20, 80, 50, 1], [50, 100, 80, 100, 1]]

station_functions = [fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal, fuzzySet.FuzzyFunctionsIdentifiers.Triangular,
                     fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal]
station_identifiers = ["close", "near", "far"]
station_parameters = [[0, 50, 0, 20, 1], [20, 80, 50, 1], [50, 100, 80, 100, 1]]


battery_fuzzy_set = fuzzySet.FuzzySet(functions=battery_functions, identifiers=battery_identifiers,
                                      parameters=battery_parameters, step=1, end=101)

target_fuzzy_set = fuzzySet.FuzzySet(functions=target_functions, identifiers=target_identifiers,
                                     parameters=target_parameters, step=1, end=101)

station_fuzzy_set = fuzzySet.FuzzySet(functions=station_functions, identifiers=station_identifiers,
                                      parameters=station_parameters, step=1, end=101)

fuzzy_set_list = [battery_fuzzy_set.get_fuzzy_functions()[0].get_fuzzy_function()[0:], battery_fuzzy_set.get_fuzzy_functions()[1].get_fuzzy_function()[0:], battery_fuzzy_set.get_fuzzy_functions()[2].get_fuzzy_function()[0:],
                  target_fuzzy_set.get_fuzzy_functions()[0].get_fuzzy_function()[0:], target_fuzzy_set.get_fuzzy_functions()[1].get_fuzzy_function()[0:], target_fuzzy_set.get_fuzzy_functions()[2].get_fuzzy_function()[0:],
                  station_fuzzy_set.get_fuzzy_functions()[0].get_fuzzy_function()[0:], station_fuzzy_set.get_fuzzy_functions()[1].get_fuzzy_function()[0:], station_fuzzy_set.get_fuzzy_functions()[2].get_fuzzy_function()[0:]]

np.savetxt("sets_v4.csv",
                       np.transpose(fuzzy_set_list),
                       delimiter=", ",
                       fmt='% s')

battery_functions = [fuzzySet.FuzzyFunctionsIdentifiers.Sigmoid, fuzzySet.FuzzyFunctionsIdentifiers.Gaussian,
                     fuzzySet.FuzzyFunctionsIdentifiers.Sigmoid]
battery_identifiers = ["low", "medium", "high"]
battery_parameters = [[30, -0.5, 0.1], [50, 20, 0.1], [70, 0.5, 0.1]]

target_functions = [fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal, fuzzySet.FuzzyFunctionsIdentifiers.Triangular,
                    fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal]
target_identifiers = ["close", "near", "far"]
target_parameters = [[0, 50, 0, 20, 1], [20, 80, 50, 1], [50, 100, 80, 100, 1]]

station_functions = [fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal, fuzzySet.FuzzyFunctionsIdentifiers.Triangular,
                     fuzzySet.FuzzyFunctionsIdentifiers.Trapezoidal]
station_identifiers = ["close", "near", "far"]
station_parameters = [[0, 50, 0, 20, 1], [20, 80, 50, 1], [50, 100, 80, 100, 1]]

battery_fuzzy_set = fuzzySet.FuzzySet(functions=battery_functions, identifiers=battery_identifiers,
                                      parameters=battery_parameters, step=1, end=101)

target_fuzzy_set = fuzzySet.FuzzySet(functions=target_functions, identifiers=target_identifiers,
                                     parameters=target_parameters, step=1, end=101)

station_fuzzy_set = fuzzySet.FuzzySet(functions=station_functions, identifiers=station_identifiers,
                                      parameters=station_parameters, step=1, end=101)

fuzzy_set_list = [battery_fuzzy_set.get_fuzzy_functions()[0].get_fuzzy_function()[0:], battery_fuzzy_set.get_fuzzy_functions()[1].get_fuzzy_function()[0:], battery_fuzzy_set.get_fuzzy_functions()[2].get_fuzzy_function()[0:],
                  target_fuzzy_set.get_fuzzy_functions()[0].get_fuzzy_function()[0:], target_fuzzy_set.get_fuzzy_functions()[1].get_fuzzy_function()[0:], target_fuzzy_set.get_fuzzy_functions()[2].get_fuzzy_function()[0:],
                  station_fuzzy_set.get_fuzzy_functions()[0].get_fuzzy_function()[0:], station_fuzzy_set.get_fuzzy_functions()[1].get_fuzzy_function()[0:], station_fuzzy_set.get_fuzzy_functions()[2].get_fuzzy_function()[0:]]

np.savetxt("sets_v5.csv",
                       np.transpose(fuzzy_set_list),
                       delimiter=", ",
                       fmt='% s')