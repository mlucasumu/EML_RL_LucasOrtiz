import numpy as np
from .base_learner import BaseLearner


class SARSA(BaseLearner):

    def __init__(self,  state_size, action_size, alpha, gamma, policy):
        super().__init__(state_size, action_size)
        self.alpha = alpha # Tasa de aprendizaje
        self.gamma = gamma # Tasa de descuento
        self.policy = policy # Política a optimizar (SARSA en on-oplicy )

    def start_episode(self):
        return

    def step(self, state, action, reward, next_state, done):
        '''
        Q(s,a) := Q(s,a) + alpha * [R + gamma * Q(s',a') - Q(s,a)]
        '''
        # Elegimos siguiente acción en base a la política del agente
        next_action = self.policy.select_action(next_state, self.qtable)
        future_q_value = (not done) * self.qtable[next_state, next_action]

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