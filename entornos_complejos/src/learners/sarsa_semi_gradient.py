import numpy as np
from .base_learner import BaseLearner


class SARSASemiGradient(BaseLearner):
    """
    SARSA Semi-gradient https://rl-anonymity-with-python.readthedocs.io/en/latest/Examples/semi_gradient_sarsa_three_columns.html

    Utiliza una función diferenciable como aproximador para representar la función de acción/valor:
        q_hat(s, a, w) = w · x(s, a)

    Regla de actualización de pesos:
        w <- w + alpha * [R + gamma * q_hat(s', a', w) - q_hat(s, a, w)] * (delta)q_hat(s, a, w)

    Para transiciones terminales el objetivo lo reduce a solamente R:
        w <- w + alpha * [R - q_hat(s, a, w)] * (delta)q_hat(s, a, w)
    """

    def __init__(self, state_size, action_size, alpha, gamma, policy, feature_fn):
        """
        state_size  : int - dimensionalidad de estados
        action_size : int - numero de acciones
        alpha       : float - learning rate
        gamma       : float - factor de descuento
        policy      : Policy - política utilizada para seleccionar acciones
        feature_fn  : callable - función diferenciable como aproximador
        """
        self.alpha = alpha
        self.gamma = gamma
        self.policy = policy
        self.d = state_size
        self.feature_fn = feature_fn

        # Inicializamos el vector de pesos w € R^d (zeros -> q_hat empieza en 0)
        self.w = np.zeros(self.d)
        self.d = self.d
        super().__init__(state_size, action_size)

    def q_hat(self, state, action):
        """Aproximación acción-valor: q_hat(s, a, w) = w · x(s, a)"""
        return float(np.dot(self.w, self.feature_fn(state, action)))

    def grad_q_hat(self, state, action):
        """
        Gradiente de q_hat
        (gradiente)_w q_hat(s, a, w) = x(s, a).
        """
        return self.feature_fn(state, action)

    def q_values(self, state):
        """devolvemos q_hat(state, ·) para todas las acciones. utilizado por la politica"""
        return np.array([self.q_hat(state, a) for a in range(self.action_size)])

    def start_episode(self):
        return

    def step(self, state, action, reward, next_state, done):
        """
        Actualización de pesos de SARSA Semi-gradient.

        If next_state is terminal (done=True):
            w <- w + alpha * [R - q_hat(s, a, w)] * (gradiente)q_hat(s, a, w)

        Else:
            a' ~ policy(s')
            w <- w + alpha * [R + gamma * q_hat(s', a', w) - q_hat(s, a, w)] * (gradiente)q_hat(s, a, w)
        """
        current_q = self.q_hat(state, action)

        if done:
            # Paso terminal
            target = reward
        else:
            # Escoger la siguiente accion utilizando la politica y el vector de pesos
            next_action = self.policy.select_action(next_state, self.q_values(next_state))
            target = reward + self.gamma * self.q_hat(next_state, next_action)

        # Error (delta)
        delta = target - current_q

        # Actualización de pesos: w <- w + alpha * delta * (gradiente)q_hat(s, a, w)
        self.w += self.alpha * delta * self.grad_q_hat(state, action)

        # cum_training_error
        self.stats['cum_training_error'] += abs(delta)

    def end_episode(self):
        return

    def reset(self):
        super().reset()
        self.w = np.zeros(self.d)
        self.stats['cum_training_error'] = 0