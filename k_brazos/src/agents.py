import numpy as np

class Agent:
    """
    Clase base para agentes.
    """
    def __init__(self, k):
        self.k = k
        self.q_est = np.zeros(k) # Estimación de valor Q(a)
        self.n_actions = np.zeros(k) # Contador de veces que se ha elegido cada acción N(a)
        self.t = 0 # Paso de tiempo

    def select_action(self):
        raise NotImplementedError

    def update(self, action, reward):
        self.n_actions[action] += 1
        self.t += 1
        # Actualización incremental de la media: Q(a) <- Q(a) + 1/N(a) * (R - Q(a))
        self.q_est[action] += (1 / self.n_actions[action]) * (reward - self.q_est[action])

    def reset(self):
        self.q_est = np.zeros(self.k)
        self.n_actions = np.zeros(self.k)
        self.t = 0

class EpsilonGreedyAgent(Agent):
    """
    Agente Epsilon-Greedy.
    """
    def __init__(self, k, epsilon=0.1):
        super().__init__(k)
        self.epsilon = epsilon

    def select_action(self):
        # Exploración
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)
        # Explotación (con desempate aleatorio)
        else:
            # np.argmax devuelve el primer índice en caso de empate, así que usamos un método más robusto
            # best_actions = np.flatnonzero(self.q_est == self.q_est.max())
            # return np.random.choice(best_actions)
            return np.argmax(self.q_est) # Simplificación común, aunque sesgada a índices menores en empates iniciales.
            # Para una implementación más rigurosa de empates:
            # return np.random.choice(np.flatnonzero(self.q_est == self.q_est.max()))

class DecayingEpsilonGreedyAgent(Agent):
    """
    Agente Epsilon-Greedy con decaimiento.
    epsilon_t = 1 / (1 + lambda * t)
    """
    def __init__(self, k, decay_rate=0.01):
        super().__init__(k)
        self.decay_rate = decay_rate

    def select_action(self):
        epsilon = 1 / (1 + self.decay_rate * self.t)
        if np.random.random() < epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.q_est)

class EpsilonFirstAgent(Agent):
    """
    Agente Epsilon-First (Explore-Then-Commit).
    Explora puramente durante los primeros epsilon * horizon pasos.
    Luego explota puramente (Greedy).
    """
    def __init__(self, k, epsilon=0.1, horizon=1000):
        super().__init__(k)
        self.epsilon = epsilon
        self.horizon = horizon
        self.exploration_steps = int(self.epsilon * self.horizon)
    
    def select_action(self):
        if self.t < self.exploration_steps:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.q_est)

class OptimisticAgent(Agent):
    """
    Agente Greedy con Valores Iniciales Optimistas.
    """
    def __init__(self, k, initial_value=5.0, alpha=0.1):
        super().__init__(k)
        self.q_est = np.ones(k) * initial_value
        self.alpha = alpha # Paso constante para olvidar el optimismo inicial gradualmente (o usar 1/N)
        # Nota: Si usamos sample-average (1/N), el valor inicial cuenta como una muestra?
        # En la implementación clásica de Sutton & Barto, se usa paso constante alpha=0.1 a veces,
        # o simplemente se empieza con Q_1 = 5 y se usa la media muestral norma.
        # Vamos a usar la media muestral estándar pero inicializada.
    
    def update(self, action, reward):
        self.n_actions[action] += 1
        self.t += 1
        # Actualización estándar
        self.q_est[action] += (1 / self.n_actions[action]) * (reward - self.q_est[action])

    def select_action(self):
        # Greedy puro (epsilon=0)
        return np.argmax(self.q_est)

class UCBAgent(Agent):
    """
    Agente Upper Confidence Bound (UCB).
    """
    def __init__(self, k, c=2):
        super().__init__(k)
        self.c = c

    def select_action(self):
        # Si hay acciones que no se han probado nunca, probarlas primero (para evitar división por cero y maximizar bound)
        if 0 in self.n_actions:
            return np.argmin(self.n_actions) # Devuelve el índice de la primera acción con count 0
        
        uncertainty = self.c * np.sqrt(np.log(self.t) / self.n_actions)
        return np.argmax(self.q_est + uncertainty)

class GradientBanditAgent(Agent):
    """
    Agente de Bandido de Gradiente.
    """
    def __init__(self, k, alpha=0.1, baseline=True):
        super().__init__(k)
        self.alpha = alpha
        self.baseline = baseline
        self.H = np.zeros(k) # Preferencias
        self.pi = np.zeros(k) # Probabilidades
        self.avg_reward = 0 # Línea base

    def softmax(self):
        exp_h = np.exp(self.H - np.max(self.H)) # Resta max para estabilidad numérica
        self.pi = exp_h / np.sum(exp_h)
        return self.pi

    def select_action(self):
        probs = self.softmax()
        return np.random.choice(self.k, p=probs)

    def update(self, action, reward):
        self.n_actions[action] += 1
        self.t += 1
        
        # Actualizar línea base (promedio de recompensas)
        if self.baseline:
            self.avg_reward += (1 / self.t) * (reward - self.avg_reward)
        else:
            self.avg_reward = 0 # Sin baseline
            
        # Actualizar preferencias H
        # H(A_t) = H(A_t) + alpha * (R_t - R_bar) * (1 - pi(A_t))
        # H(a) = H(a) - alpha * (R_t - R_bar) * pi(a)  para a != A_t
        
        one_hot = np.zeros(self.k)
        one_hot[action] = 1
        
        baseline_term = (reward - self.avg_reward)
        self.H += self.alpha * baseline_term * (one_hot - self.pi)
        
    def reset(self):
        super().reset()
        self.H = np.zeros(self.k)
        self.avg_reward = 0
class UCB2Agent(Agent):
    """
    Agente UCB2 (Auer et al. 2002).
    """
    def __init__(self, k, alpha=0.1):
        super().__init__(k)
        self.alpha = alpha
        self.r = np.zeros(k, dtype=int) # Número de épocas por brazo
        self.cur_action = -1 # Acción actual ejecutándose en la época
        self.epoch_steps_left = 0 # Pasos restantes en la época actual

    def __tau(self, r):
        return int(np.ceil((1 + self.alpha) ** r))

    def select_action(self):
        # Si estamos en medio de una época, continuar con la misma acción
        if self.epoch_steps_left > 0:
            self.epoch_steps_left -= 1
            return self.cur_action

        # Seleccionar nueva acción para la siguiente época
        # Asegurar que n > 0 para evitar log(0) o división por cero
        n = max(1, self.t)
        
        # Calcular bounds
        # a_{n, r} = sqrt( (1+alpha) * ln(e * n / tau(r)) / (2 * tau(r)) )
        
        best_val = -float('inf')
        best_arm = -1
        
        for k in range(self.k):
            tau_r = self.__tau(self.r[k])
            
            # Evitar división por cero si tau_r es 0 (no debería ocurrir con ceil y >=0)
            if tau_r == 0: tau_r = 1
                
            term1 = (1 + self.alpha) * np.log(np.e * n / tau_r)
            term2 = 2 * tau_r
            a_nr = np.sqrt(term1 / term2)
            
            val = self.q_est[k] + a_nr
            if val > best_val:
                best_val = val
                best_arm = k

        self.cur_action = best_arm
        
        # Calcular duración de la época: tau(r+1) - tau(r)
        current_r = self.r[self.cur_action]
        duration = self.__tau(current_r + 1) - self.__tau(current_r)
        self.epoch_steps_left = duration - 1 # Restamos 1 porque ya consumimos este paso
        
        # Incrementar contador de épocas para este brazo
        self.r[self.cur_action] += 1
        
        return self.cur_action
        
    def reset(self):
        super().reset()
        self.r = np.zeros(self.k, dtype=int)
        self.cur_action = -1
        self.epoch_steps_left = 0

class SoftmaxAgent(Agent):
    """
    Agente con selección de acción Softmax (Boltzmann Exploration).
    Probabilidad de elegir acción a: e^(Q(a)/tau) / sum(e^(Q(b)/tau))
    """
    def __init__(self, k, tau=0.1):
        super().__init__(k)
        self.tau = tau

    def select_action(self):
        # Estabilidad numérica: restar max(Q)
        exp_q = np.exp((self.q_est - np.max(self.q_est)) / self.tau)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(self.k, p=probs)
