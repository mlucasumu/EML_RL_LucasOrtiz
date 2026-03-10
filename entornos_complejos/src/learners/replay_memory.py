import random
from collections import deque

class ReplayMemory:
    """
    Memoria de repetición (Experience Replay) de capacidad N.
    Almacena transiciones (phi_t, a_t, r_t, phi_{t+1}, done) y permite
    muestrear minibatches de forma aleatoria para romper correlaciones temporales.
    """

    def __init__(self, capacity):
        # Usamos deque con maxlen para descartar automáticamente las transiciones más antiguas
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Almacena una nueva transición en la memoria."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Devuelve un minibatch aleatorio de transiciones."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)