import gymnasium as gym
import random
import numpy as np

def make_tile_feature_fn(bins, n_tilings, n_actions):
    """
    Construye la feature_fn compatible con SARSASemiGradient.
    
    El vector de features x(s, a) es binario y disperso:
    - n_tilings bloques, uno por tiling
    - Dentro de cada bloque, n_actions sub-bloques
    - Solo el tile activo en cada tiling se activa (= 1.0)
    
    Dimensión total: n_tilings x prod(bins) x n_actions
    """
    tiles_per_tiling = int(np.prod(bins)) # 8^4 = 4096
    block_size = tiles_per_tiling * n_actions # 4096 × 2 = 8192
    total_dim = n_tilings * block_size # 4 × 8192 = 32768

    def flat_tile_index(tiling_indices, bins):
        """Convierte una tupla de índices a un único índice plano (row-major)."""
        idx = 0
        for i, (ti, b) in enumerate(zip(tiling_indices, bins)):
            # Clamp para evitar desbordamientos por np.digitize
            ti_clamped = int(np.clip(ti, 0, b - 1))
            idx = idx * b + ti_clamped
        return idx

    def feature_fn(state, action):
        """
        state  : lista de 4 tuplas [(d0,d1,d2,d3), ...], una por tiling
        action : int  (0 ó 1 en CartPole)
        
        Devuelve vector binario de longitud total_dim.
        """
        x = np.zeros(total_dim)

        for t_idx, tiling_indices in enumerate(state):
            tile_idx   = flat_tile_index(tiling_indices, bins)
            # Offset: bloque del tiling + sub-bloque de la acción
            offset = t_idx * block_size + action * tiles_per_tiling + tile_idx
            x[offset]  = 1.0

        return x

    return feature_fn, total_dim

class TileCodingEnv(gym.ObservationWrapper):
    """    
    CartPole Observation Space (4 dimensiones):
    - Cart Position:        [-4.8,   4.8]
    - Cart Velocity:        [-inf,   inf]  -> acotado a [-3.0, 3.0]
    - Pole Angle:           [-0.418, 0.418] radianes (+-24°)
    - Pole Angular Velocity:[-inf,   inf]  -> acotado a [-3.0, 3.0]
    """

    def __init__(self, env, bins, low, high, n=4):
        """
        Parámetros:
        - env:  Entorno original de gymnasium.
        - bins: Array con el número de intervalos por dimensión.  Shape: (4,)
        - low:  Array con las cotas inferiores de cada dimensión.    Shape: (4,)
        - high: Array con las cotas superiores de cada dimensión.    Shape: (4,)
        - n:    Número de tillings (default 4).
        
        Si usamos CartPole con bins=[8,8,8,8] y n=4 tilings:
        -> el vector observacional tiene 4 dimensiones x 4 tilings = 16 componentes
        -> cada componente es un índice entero comprendido entre [0, bins[d]]
        """
        super().__init__(env)
        self.low = low
        self.high = high
        self.tilings = self._create_tilings(bins, high, low, n)
        # MultiDiscrete: un valor discreto por pareja (dimension x tiling)
        # ejemplo básico: bins=[8,8,8,8], n=4 -> nvec tiene 16 entradas, cada una en el rango [0,8]
        self.observation_space = gym.spaces.MultiDiscrete(nvec=bins.tolist() * n)

    def observation(self, obs):  # Es necesario sobreescribir este método de ObservationWrapper
        """
        Transforma una observación continua en una representación discreta usando tile coding.

        Parámetro:
        - obs: observación continua proveniente del entorno.

        Para cada tiling (rejilla) en self.tilings, se determina el índice del tile en el que
        cae cada componente de la observación mediante np.digitize. Se devuelve una lista de
        tuplas de índices, una por cada tiling.
        """
        indices = []  # Lista que almacenará los índices discretizados para cada tiling.
        for t in self.tilings:
            # Para cada tiling 't', se calcula el índice en el que se encuentra cada componente de la observación.
            tiling_indices = tuple(np.digitize(i, b) for i, b in zip(obs, t))
            indices.append(tiling_indices)  # Se agrega la tupla de índices correspondiente a la tiling actual.
        return indices  # Retorna la lista de índices de todas las tilings.

    def _create_tilings(self, bins, high, low, n):
        """
        Crea n offset tilings para el CartPole de estados de dimensión 4D

        Para 4 dimensiones: displacement_vector = [1,3,5,7]
        Esto garantiza que cada tiling se desplace de manera diferente en cada dimensión.

        Parameters:
        - bins: [n_cart_pos, n_cart_vel, n_pole_angle, n_pole_ang_vel]
        - high: [max_cart_pos, max_cart_vel, max_pole_angle, max_pole_ang_vel]
        - low:  [min_cart_pos, min_cart_vel, min_pole_angle, min_pole_ang_vel]
        - n:    Number of tilings

        Returns:
        - tilings: lista de n tilings. Cada tiling es una lista de 4 arrays de buckets
        """
        # Para 4 dims: np.arange(1, 2*4, 2) -> [1, 3, 5, 7]
        # Dim 0 (cart_pos)       desplazar en multiplos de 1
        # Dim 1 (cart_vel)       desplazar en multiplos de 3
        # Dim 2 (pole_angle)     desplazar en multiplos de 5
        # Dim 3 (pole_ang_vel)   desplazar en multiplos de 7
        displacement_vector = np.arange(1, 2 * len(bins), 2)  # [1, 3, 5, 7]

        tilings = []
        for i in range(1, n + 1):
            # De forma aleatoria se perturban los bordes ligeramente (+-20%) para evitar los artefactos de los bordes
            low_i  = low  - random.random() * 0.2 * np.abs(low)
            high_i = high + random.random() * 0.2 * np.abs(high)

            # Patrón del offset para el tilling i (mod n mantiene valores en el rango de [0, n-1]):
            # i=1: [1,3,5,7] % 4 = [1,3,1,3]
            # i=2: [2,6,10,14] % 4 = [2,2,2,2]
            # i=3: [3,9,15,21] % 4 = [3,1,3,1]
            # i=4: [4,12,20,28] % 4 = [0,0,0,0]
            displacements = displacement_vector * i % n

            # Escalamos los offsets a valores en unidades reales: una unidad = tamaño_segmentpo/n
            segment_sizes = (high_i - low_i) / bins
            displacements = displacements * (segment_sizes / n)

            low_i  += displacements
            high_i += displacements

            # Creamos los bin edges para cada una de las 4 dimensiones
            # np.linspace con (bins[d]-1) puntos crea bins[d] intervalso.
            buckets_i = [np.linspace(j, k, l - 1)
                         for j, k, l in zip(low_i, high_i, bins)]
            tilings.append(buckets_i)

        return tilings