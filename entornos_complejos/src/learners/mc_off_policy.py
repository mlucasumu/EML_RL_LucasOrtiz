import numpy as np
from .base_learner import BaseLearner


class MCOffPolicy(BaseLearner):
    def __init__(self, state_size, action_size, gamma, behavior_policy):
        super().__init__(state_size, action_size)
        self.gamma = gamma
        self.behavior_policy = behavior_policy  # Política de comportamiento b con cobertura pi. Asumimos que pi es greedy
        self.C = np.zeros([state_size, action_size])  # Pesos acumulativos de muestreo de importancia

    def start_episode(self):
        self.episode = []

    def step(self, state, action, reward, next_state, done):
        self.episode.append((state, action, reward))

    def end_episode(self):
        G = 0
        W = 1.0
        for (state, action, reward) in reversed(self.episode):
            if W == 0:
                break
            G = self.gamma * G + reward # Actualizar el retorno acumulado
            self.C[state, action] += W # Actualizar el peso acumulado para S_t
            delta = G - self.qtable[state, action]
            self.qtable[state, action] += (W / self.C[state, action]) * delta
            self.stats['cum_training_error'] += delta
            # W <- W * pi(A_t|S_t) / b(A_t|S_t)
            # pi debe de ser greedy, asisque pi(a|s) = 1 if a == argmax Q(s,·), else 0
            greedy_action = np.argmax(self.qtable[state])
            # pi(a|s) = 1, entonces W = W / b(a|s)
            b_prob = self.behavior_policy.action_probability(state, action, self.qtable)
            if action != greedy_action or b_prob == 0:
                break  # W sería 0, no tiene sentido continuar
            W = W / b_prob

    def reset(self):
        super().reset()
        self.C = np.zeros([self.state_size, self.action_size])
        self.stats['cum_training_error'] = 0