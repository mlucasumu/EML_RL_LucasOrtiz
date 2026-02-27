import numpy as np
from .base_learner import BaseLearner


class MCOffPolicy(BaseLearner):

    def __init__(self, state_size, action_size, discount_factor):
        pass

    def start_episode(self):
        pass

    def step(self, state, action, reward, next_state, done):
        pass

    def end_episode(self):
        pass

    def reset(self):
        pass