import numpy as np
from base_learner import BaseLearner


class MCOnPolicy(BaseLearner):

    def __init__(self,  state_size, action_size):
        pass

    def start_episode(self):
        pass # self.episode = []

    def step(self, state, action, reward, next_state, done):
        pass # self.episode.append((s, a, r))

    def end_episode(self):
        pass # Actualizar qtable recorriendo el episodio de atrás hacia delante