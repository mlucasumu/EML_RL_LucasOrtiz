'''
La política decide qué acción tomar en cada paso.
'''

from abc import ABC, abstractmethod

class BasePolicy(ABC):

    @abstractmethod
    def select_action(self, state, qtable):
        '''
        Dado un estado y una tabla de valores Q, devuelve la siguiente acción a realizar.
        '''
        pass

    @abstractmethod
    def action_probability(self, state, action, qtable):
        """
        Devuelve la probabilidad de que esta política elija la `action` en el `state` actual.
        """
        pass