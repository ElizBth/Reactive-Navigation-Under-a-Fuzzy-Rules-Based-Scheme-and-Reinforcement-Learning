#
# This class contains the methods and functions used in FQL
# By Elizabeth Lopez
#

from fuzzy import FuzzyInferenceSystem
import numpy as np
import random
from datetime import datetime


class FQL:
    accumulated_reward = 0
    _gamma = 0.8
    _lambda = 0.7

    def __init__(self, learning_rate, discount_factor, actions, output_functions, fuzzy_sets, fuzzy_sets_identifiers):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.actions = actions
        self.fis = FuzzyInferenceSystem.FIS(fuzzy_sets=fuzzy_sets, fuzzy_sets_identifiers=fuzzy_sets_identifiers)
        self.fis.set_output_functions(output_functions)
        self.eligibility = np.zeros((len(self.fis.get_fis_rules()), actions), dtype=float)
        self.q_table = np.zeros((len(self.fis.get_fis_rules()), actions), dtype=float)
        self.accumulated_reward = 0

    def __update_eligibility(self, rule, action):
        action_selected = np.max(self.q_table[rule])
        alpha = self.fis.get_alpha_of_rule(rule) / self.fis.get_global_alpha()

        if action_selected == action:
            eligibility = self._lambda * self._gamma * self.eligibility[rule][action] + alpha
        else:
            eligibility = self._lambda * self._gamma * self.eligibility[rule][action]

        self.eligibility[rule][action] = 0 if np.isnan(eligibility) else eligibility

    def __update_q_value(self, rule, action, new_state):
        new_q = self.learning_rate * self.__get_delta_q(new_state, action) * self.eligibility[rule][action]
        self.q_table[rule][action] = max(0, self.q_table[rule][action]) if np.isnan(new_q) or np.isinf(new_q) \
            else new_q

    def __get_delta_q(self, new_state, action):
        return self.get_reward(new_state, action) + self.discount_factor * self.__get_value_of_states() - \
               self.__get_q_value(action)

    def __get_q_value(self, action):
        q_val = 0
        for index in range(len(self.fis.get_alpha_values())):
            q_val += self.fis.get_alpha_values()[index] * self.fis.get_output_function(action) * \
                     self.q_table[index][action] / self.fis.get_global_alpha()
        return q_val

    def __get_value_of_states(self):
        return np.sum(self.fis.get_alpha_values() * np.amax(self.q_table, 1)) / self.fis.get_global_alpha()

    def get_reward(self, rule, action):
        print("Reward in rule " + str(rule) + " when action is " + str(action))
        return 1

    def step_execution(self, rule, trial):
        action = self.__get_action(rule, trial)
        new_state = self.task_execution(action)
        self.__update_eligibility(rule, action)
        self.__update_q_value(rule, action, new_state)
        self.accumulated_reward += self.get_reward(new_state, action)
        return new_state

    def __get_action(self, rule, trial):
        random.seed(datetime.now())
        exploration_probability = 10/(10+trial)
        return random.randint(0, self.actions-1) if exploration_probability > 0.5 else np.argmax(self.q_table[rule])

    def task_execution(self, action):
        print("Action to be executed " + str(action))
        return action




