import itertools
import numpy as np


class FIS:
    rules = []
    _output_function = []
    _output = []
    _fuzzy_sets = []
    _alpha = []
    _global_alpha = 0

    def __init__(self, fuzzy_sets, fuzzy_sets_identifiers):
        self._fuzzy_sets = fuzzy_sets
        self.__set_fuzzy_rules(fuzzy_sets_identifiers)
        self.set_alpha_values()

    #
    # This method is used to generate FIS rules
    #
    def __set_fuzzy_rules(self, fuzzy_sets_identifiers):
        self.rules = [p for p in itertools.product(*fuzzy_sets_identifiers)]

    #
    # This method gets rules generated
    #
    def get_fis_rules(self):
        return self.rules

    #
    # This method returns the rule number
    #
    def get_fis_rule_number(self, current_states):
        return self.rules.index(tuple(current_states))

    #
    # This method sets output's functions
    #
    def set_output_functions(self, functions):
        self._output_function = functions

    #
    # This method appends a functions on output_function list
    #
    def set_output_function(self, function):
        self._output_function.append(function)

    #
    # This method return the output function on index
    #
    def get_output_function(self, index):
        return self._output_function[index]

    #
    # This method appends an output on outputs list
    #
    def set_output(self, value):
        self._output.append(value)

    #
    # This method returns the output in index from output list
    #
    def get_output(self, index):
        return self._output[index]

    #
    # This method returns the number of the rule which corresponds to the entries on fuzzy_values
    # fuzzy_values -> list of identifier entries
    #
    def get_rule_index(self, fuzzy_values):
        status = []
        for index in range(len(self._fuzzy_sets)):
            status.append(self._fuzzy_sets[index].get_identifier_from_value(fuzzy_values[index]))
        return self.get_fis_rule_number(status)

    def get_label(self, fuzzy_set, value):
        return self._fuzzy_sets[fuzzy_set].get_identifier_from_value(value)
    #
    # This method sets the truth values of the rules on the list _alpha and also it sets the global_alpha
    #
    def set_alpha_values(self):
        average_values = []
        for set_index in range(len(self._fuzzy_sets)):
            average_values_on_sets = []
            for index in range(self._fuzzy_sets[set_index].get_number_of_functions()):
                average_values_on_sets.append(self._fuzzy_sets[set_index].get_fuzzy_set_average())
            average_values.append(average_values_on_sets)

        self._alpha = [np.prod(p) for p in itertools.product(*average_values)]
        self._global_alpha = np.sum([np.prod(p) for p in itertools.product(*average_values)])

    #
    # This method returns global alpha
    #
    def get_global_alpha(self):
        return self._global_alpha

    #
    # This method return the alpha (truth value) in the rule index
    #
    def get_alpha_of_rule(self, index):
        return self._alpha[index]

    #
    # This method returns the list with all alpha values
    #
    def get_alpha_values(self):
        return self._alpha




