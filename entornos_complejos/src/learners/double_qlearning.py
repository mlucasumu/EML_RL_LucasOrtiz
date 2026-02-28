import numpy as np
from .base_learner import BaseLearner


class DoubleQLearning(BaseLearner):

    def __init__(self, state_size, action_size, alpha, gamma):
        super().__init__(state_size, action_size)
        self.alpha = alpha # Tasa de aprendizaje
        self.gamma = gamma # Tasa de descuento
        self.turn = 1 # Qué tabla toca actualizar

    #@property
    #def qtable(self):
    #    return self.Q1 + self.Q2

    def start_episode(self):
        return

    def step(self, state, action, reward, next_state, done):
        '''
        Q1(s,a) := Q1(s,a) + alpha * [R + gamma * max Q2(s',argmax Q1(s',a)) - Q1(s,a)] y viceversa
        Elegimos acción futura con una tabla pero evaluamos con otra.
        '''
        if self.turn == 1:
            # Actualizamos Q1
            best_action = np.argmax(self.Q1[next_state])
            target = reward + self.gamma * self.Q2[next_state, best_action] * (not done)
            delta = target - self.Q1[state, action]
            self.Q1[state, action] += self.alpha * delta
            self.turn = 2
        else:
            # Actualizamos Q2
            best_action = np.argmax(self.Q2[next_state])
            target = reward + self.gamma * self.Q1[next_state, best_action] * (not done)
            delta = target - self.Q2[state, action]
            self.Q2[state, action] += self.alpha * delta
            self.turn = 1

        self.qtable[state, action] += self.alpha * delta # qtable es la suma de Q1 y Q2

        # Loggear estadísticas
        self.stats['cum_training_error'] += delta

    def end_episode(self):
        return
    
    def reset(self):
        super().reset()
        self.Q1 = np.zeros((self.state_size, self.action_size))
        self.Q2 = np.zeros((self.state_size, self.action_size))
        self.stats['cum_training_error'] = 0