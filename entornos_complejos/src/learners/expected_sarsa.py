from .base_learner import BaseLearner


class ExpectedSARSA(BaseLearner):

    def __init__(self, state_size, action_size, alpha, gamma, policy):
        super().__init__(state_size, action_size)
        self.alpha = alpha # Tasa de aprendizaje
        self.gamma = gamma # Tasa de descuento
        self.policy = policy # Política a optimizar

    def start_episode(self):
        return

    def step(self, state, action, reward, next_state, done):
        '''
        Q(s,a) := Q(s,a) + alpha * [R + gamma * E[Q(s',a')] - Q(s,a)]
        '''
        expected_q_value = (not done) * self.policy.expected_value(next_state, self.qtable)

        delta = (
            reward
            + self.gamma * expected_q_value
            - self.qtable[state, action]
        )
        self.qtable[state, action] = self.qtable[state, action] + self.alpha * delta

        # Loggear estadísticas
        self.stats['cum_training_error'] += abs(delta)

    def end_episode(self):
        return
    
    def reset(self):
        super().reset()
        self.stats['cum_training_error'] = 0