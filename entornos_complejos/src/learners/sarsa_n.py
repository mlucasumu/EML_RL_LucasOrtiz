from collections import deque
from .base_learner import BaseLearner


class nStepSARSAonPolicy(BaseLearner):

    def __init__(self,  state_size, action_size, alpha, gamma, policy, n):
        super().__init__(state_size, action_size)
        self.alpha = alpha # Tasa de aprendizaje
        self.gamma = gamma # Tasa de descuento
        self.policy = policy # Política a optimizar
        self.n = n # Número de pasos

        self.buffer = deque() # Para almacenar los n pasos

    def start_episode(self):
        self.buffer.clear()

    def step(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward))

        if len(self.buffer) >= self.n:
            self.update(next_state, done)

        # Si llegamos al final, actualizamos hasta vaciar el buffer
        if done:
            while self.buffer:
                self.update(next_state=None, done=True)

    def update(self, next_state, done):
        pass # TODO: implementar esto (en sutton barto, sería la parte de tau >= 0)

    def end_episode(self):
        self.buffer.clear() # ¿necesario?