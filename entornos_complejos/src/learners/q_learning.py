import numpy as np
from .base_learner import BaseLearner


class QLearning(BaseLearner):

    def __init__(self,  state_size, action_size, alpha, gamma):
        super().__init__(state_size, action_size)
        self.alpha = alpha # Tasa de aprendizaje
        self.gamma = gamma # Tasa de descuento

    def start_episode(self):
        return

    def step(self, state, action, reward, next_state, done):
        '''
        Q(s,a) := Q(s,a) + alpha * [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        '''
        future_q_value = (not done) * np.max(self.qtable[next_state, :])
        delta = (
            reward
            + self.gamma * future_q_value
            - self.qtable[state, action]
        )
        self.qtable[state, action] = self.qtable[state, action] + self.alpha * delta

        # Loggear estadísticas
        self.stats['cum_training_error'] += delta

    def end_episode(self):
        return
    
    def reset(self):
        super().reset()
        self.stats['cum_training_error'] = 0
