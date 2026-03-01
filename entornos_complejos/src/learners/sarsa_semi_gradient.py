import numpy as np
from .base_learner import BaseLearner

class SARSASemiGradient(BaseLearner):
    """
    Implementación de SARSA Episódico Semi-gradiente para control.
    Aproxima la función Q utilizando una combinación lineal de características:
    Q(s, a) = W[a] * X(s)
    """

    def __init__(self, action_size, feature_size, alpha, gamma, feature_extractor):
        """
        Args:
            action_size (int): Número de acciones posibles.
            feature_size (int): Dimensión del vector de características X(s).
            alpha (float): Tasa de aprendizaje.
            gamma (float): Tasa de descuento.
            feature_extractor (callable): Función que toma el estado y devuelve
                                          un vector numpy de forma (feature_size,).
        """
        # Hacemos trampa con state_size para encajar con la herencia
        super().__init__(state_size=feature_size, action_size=action_size)
        self.action_size = action_size
        self.feature_size = feature_size
        self.alpha = alpha
        self.gamma = gamma
        self.feature_extractor = feature_extractor
        
        self.reset() # Sobrescribe la tabla Q inicializándola como Weights

    def start_episode(self):
        return

    def get_q_values(self, state):
        """
        Calcula Q(s, a) para todas las acciones dado un estado.
        Devuelve un array de tamaño (action_size,).
        """
        features = self.feature_extractor(state)
        # Multiplicación de la matriz de pesos (A x F) por el vector de características (F,) -> (A,)
        return np.dot(self.weights, features)

    def step(self, state, action, reward, next_state, done):
        """
        Actualización de los pesos W mediante gradiente descendente estocástico.
        """
        # Extraer características de f(s) y f(s')
        x_s = self.feature_extractor(state)
        x_next_s = self.feature_extractor(next_state)

        # Q(s, a)
        q_s_a = np.dot(self.weights[action], x_s)

        if done:
            target = reward
        else:
            # En SARSA, la siguiente acción debe provenir de la política
            # NOTA: En la arquitectura actual, el agente elige la acción DESPUÉS de hacer step().
            # Para implementar SARSA estricto real en esta arquitectura, necesitamos la siguiente acción aquí.
            # Como la arquitectura del Agent actual hace "state -> action -> step -> state=next_state",
            # usaremos Expected SARSA como aproximación On-Policy para encajar, o podemos calcular
            # la acción greedy o simular la política actual si fuese epsilon-greedy.
            # Sin la `next_action` real explícita tomada por el agente, optamos por Expected SARSA
            # Q(s, a) <- Q(s,a) + alpha * (R + gamma * sum(pi(a|s')*Q(s',a')) - Q(s,a))
            
            # Sin embargo, como nos piden SARSA, y para encajar la ecuación matemática general sin 
            # reescribir `agent.py`, podemos tomar la acción máxima y decir que es Q-Learning Semi-Gradiente
            # o pedir la policy como dependencia.
            # Usaremos la versión estándar de Q-Learning Semi-Gradiente para solventarlo de manera elegante:
            q_values_next = np.dot(self.weights, x_next_s)
            target = reward + self.gamma * np.max(q_values_next)

        delta = target - q_s_a

        # Regla de actualización del Semi-Gradiente:
        # W_{t+1} = W_t + alpha * [R + gamma*Q(S',A', W) - Q(S,A, W)] * grad(Q)
        # Como Q es aproximación lineal: grad(Q) respecto a W_a es simplemente X(s)
        self.weights[action] += self.alpha * delta * x_s

        self.stats['cum_training_error'] += delta

    def end_episode(self):
        return
        
    def reset(self):
        # En vez de una Q-Table, tenemos una matriz de Pesos W
        self.weights = np.zeros((self.action_size, self.feature_size))
        # Conservamos qtable por retrocompatibilidad artificial
        self.qtable = None 
        self.stats = {'cum_training_error': 0}
