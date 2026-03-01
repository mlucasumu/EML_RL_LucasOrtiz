from collections import deque
from .base_learner import BaseLearner


class nStepSARSAonPolicy(BaseLearner):

    def __init__(self, state_size, action_size, alpha, gamma, policy, n):
        super().__init__(state_size, action_size)
        self.alpha = alpha # Tasa de aprendizaje
        self.gamma = gamma # Tasa de descuento
        self.policy = policy # Política a optimizar
        self.n = n # Número de pasos

        self.buffer = deque() # Para almacenar los n pasos

    def start_episode(self):
        self.buffer.clear()

    def step(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward)) # Añadimos al final del buffer

        if len(self.buffer) >= self.n:
            self.update(next_state, done)

        # Si llegamos al final, actualizamos hasta vaciar el buffer
        if done:
            while self.buffer:
                self.update(next_state=None, done=True)

    def update(self, next_state, done):
        # En Sutton Barto, sería el condicional tau >= 0
        G = 0.0
        gamma = 1.0 # Actualizamos el gamma paso a paso para optimizar (en lugar de hacer gamma**i)

        for i, (_, _, r) in enumerate(self.buffer):
            G += gamma * r
            gamma *= self.gamma

        # Bootstrap si no es terminal y tenemos n pasos completos
        if not done and len(self.buffer) == self.n:
            next_action = self.policy.select_action(next_state, self.qtable) # tau = t - n + 1. Necesitamos el Q valor de tau + n = t + 1 -> siguiente acción
            G += gamma * self.qtable[next_state, next_action]

        s, a, _ = self.buffer.popleft() # Sacamos el primer elemento del buffer para actualizar su valor Q 

        delta = G - self.qtable[s, a]
        self.qtable[s, a] += self.alpha * delta
        self.stats['cum_training_error'] += abs(delta)

    def end_episode(self):
        self.buffer.clear()

    def reset(self):
        super().reset()
        self.stats['cum_training_error'] = 0