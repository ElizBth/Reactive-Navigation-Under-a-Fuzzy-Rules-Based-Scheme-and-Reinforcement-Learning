import numpy as np
import random


class QL:
    def __init__(self, total_actions, total_states, learning_rate, discount_factor, exploration_rate=0.5):
        self.q_table = np.zeros((len(total_states), total_actions), dtype=float)
        self.reward = []
        self.states = total_states
        self.actions = []
        self.total_actions = total_actions
        self.accumulated_reward = 0
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def __select_action(self, trial, state):
        probability = 10/10+trial
        if probability < self.exploration_rate:
            action = np.argmax(self.q_table[state])
        else:
            action = random.randint(0, self.total_actions - 1)
        return action

    def get_state(self, value):
        return self.states.index(value)

    def get_reward(self, action, state):
        return 0.1

    def task_execution(self, action):
        print("Action to be executed " + str(action))
        return action

    def step_execution(self, state, trial):
        action = self.__select_action(state, trial)
        new_state = self.task_execution(action)
        reward = self.get_reward(action, new_state)
        best_q = np.max(self.q_table[new_state])
        new_q_value = self.q_table[state, action] + self.learning_rate * (reward + self.discount_factor * best_q -
                                                                          self.q_table[state, action])
        self.reward.append(reward)
        # self.states.append(new_state)
        self.actions.append(action)

        #print(" new_q_value " + str(new_q_value))
        #if not np.isnan(new_q_value) and np.isinf(new_q_value):
        self.q_table[state, action] = new_q_value
        return new_state, action, reward
