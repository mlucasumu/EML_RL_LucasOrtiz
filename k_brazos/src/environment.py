import numpy as np

class Bandit:
    """
    Entorno del Bandido de k-brazos.
    """
    def __init__(self, k, mu=0, sigma=1, stationary=True, distribution='normal', n_binomial=10):
        """
        Inicializa el bandido.
        
        Args:
            k (int): Número de brazos.
            mu (float): Media (o p para bernoulli/binomial) de la distribución.
            sigma (float): Desviación estándar (solo para normal).
            stationary (bool): Si es True, las medias de los brazos son fijas.
            distribution (str): 'normal', 'bernoulli', o 'binomial'.
            n_binomial (int): Número de intentos para distribución binomial.
        """
        self.k = k
        self.mu = mu
        self.sigma = sigma
        self.stationary = stationary
        self.distribution = distribution.lower()
        self.n_binomial = n_binomial
        
        # Inicializar las medias reales de cada brazo (q*(a))
        if self.distribution == 'normal':
            self.q_star = np.random.normal(self.mu, 1, self.k)
        elif self.distribution == 'bernoulli':
            # p aleatoria entre 0 y 1 para cada brazo
            self.q_star = np.random.random(self.k)
        elif self.distribution == 'binomial':
            # p aleatoria entre 0 y 1 para cada brazo, valor esperado n*p
            self.probs = np.random.random(self.k)
            self.q_star = self.n_binomial * self.probs
        
        self.best_action = np.argmax(self.q_star)

    def step(self, action):
        """
        Ejecuta una acción y devuelve la recompensa.
        
        Args:
            action (int): Índice del brazo a jalar (0 a k-1).
            
        Returns:
            reward (float): Recompensa obtenida.
        """
        if self.distribution == 'normal':
            return np.random.normal(self.q_star[action], self.sigma)
        elif self.distribution == 'bernoulli':
            # q_star contiene las probabilidades p
            return np.random.binomial(1, self.q_star[action])
        elif self.distribution == 'binomial':
            # q_star contiene n*p, pero necesitamos p almacenado en self.probs
            return np.random.binomial(self.n_binomial, self.probs[action])
        else:
            raise ValueError(f"Distribución desconocida: {self.distribution}")

    def reset(self):
        """
        Reinicia el entorno, generando nuevas medias para los brazos.
        """
        if self.distribution == 'normal':
            self.q_star = np.random.normal(self.mu, 1, self.k)
        elif self.distribution == 'bernoulli':
            self.q_star = np.random.random(self.k)
        elif self.distribution == 'binomial':
            self.probs = np.random.random(self.k)
            self.q_star = self.n_binomial * self.probs
            
        self.best_action = np.argmax(self.q_star)


