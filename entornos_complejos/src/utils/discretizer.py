import numpy as np
import gymnasium as gym

class StateDiscretizer:
    """
    Clase utilitaria para convertir los espacios de observación continuos de Gymnasium
    (como MountainCar) en estados discretos indexables para métodos tabulares.
    """
    def __init__(self, env: gym.Env, bins: tuple):
        """
        Inicializa el discretizador.

        Args:
            env (gym.Env): El entorno continuo de Gymnasium.
            bins (tuple): Una tupla con el número de "cajas" (bins) deseadas 
                          para cada dimensión del estado. 
                          Ejemplo para MountainCar: (20, 20)
        """
        self.env = env
        self.bins = bins
        self.lower_bounds = env.observation_space.low
        self.upper_bounds = env.observation_space.high
        
        # Para MountainCar, la velocidad limite es infinito. Lo acotamos a mano.
        if np.any(self.upper_bounds == np.inf):
           self.upper_bounds = np.ones_like(self.upper_bounds)*10
        if np.any(self.lower_bounds == -np.inf):
           self.lower_bounds = np.ones_like(self.lower_bounds)*-10

        self.bin_widths = (self.upper_bounds - self.lower_bounds) / np.array(self.bins)

    def get_state_size(self):
        """Devuelve el tamaño total de la tabla (S_1 * S_2 * ... * S_N)"""
        return np.prod(self.bins)

    def discretize(self, state: np.ndarray) -> int:
        """
        Convierte una matriz/vector de estado continuo en un único entero 
        que representa el índice "aplanado" (flattened) de la caja correspondiente.
        """
        # Calcular el índice por cada dimensión
        ratios = (state - self.lower_bounds) / self.bin_widths
        indices = np.floor(ratios).astype(int)
        
        # Asegurarse de que no nos salimos de los límites
        indices = np.clip(indices, 0, np.array(self.bins) - 1)
        
        # Convertir un índice N-dimensional (ej: [pos_x, vel_v]) a un único escalar 1D
        # np.ravel_multi_index hace exactamente esto basándose en las dimensiones (bins)
        state_int = np.ravel_multi_index(indices, self.bins)
        return state_int
