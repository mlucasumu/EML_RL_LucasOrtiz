'''
La política decide qué acción tomar en cada paso.
'''

from abc import ABC, abstractmethod
import numpy as np

class BasePolicy(ABC):

    @abstractmethod
    def select_action(self, state, qtable):
        '''
        Dado un estado y una tabla de valores Q, devuelve la siguiente acción a realizar.
        '''
        pass

    @abstractmethod
    def action_probabilities(self, state, qtable):
        '''
        Calcula las probabilidades de seleccionar cada acción en el estado state.
        '''
        pass

    def action_probability(self, state, action, qtable):
        probs = self.action_probabilities(state, qtable)
        return probs[action]
    
    def expected_value(self, state, qtable):
        probs = self.action_probabilities(state, qtable)
        return np.dot(probs, qtable[state])