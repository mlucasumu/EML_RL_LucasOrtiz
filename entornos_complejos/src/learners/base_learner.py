'''
El Learner es el algoritmo de aprendizaje, por lo que decide cómo y 
cuándo actualizar la política/tabla de valores Q.
'''

from abc import ABC, abstractmethod
import numpy as np


class BaseLearner(ABC):

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.reset()
        
    @abstractmethod
    def start_episode(self):
        '''
        Inicializa estructuras de datos necesarias para el aprendizaje en un episodio.
        '''
        pass

    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        '''
        LLeva a cabo la actualización tras un paso.
        '''
        pass

    @abstractmethod
    def end_episode(self):
        '''
        Limpia estructuras de datos y realiza actualizaciones finales en caso de que sea necesario.
        '''
        pass

    def reset(self):
        '''
        Resetea el algoritmo de aprendizaje para una nueva ejecución.
        '''
        self.qtable = np.zeros((self.state_size, self.action_size))
        self.stats = {} # Estadísticas particulares de cada algoritmo (p.ej. training error en TD)