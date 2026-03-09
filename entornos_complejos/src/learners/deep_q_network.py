import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

#from .q_network import QNetwork
from .base_learner import BaseLearner


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


class DQNLearner(BaseLearner):
    """
    Learner que implementa Deep Q-Learning con Experience Replay
    según el Algoritmo 1 de Mnih et al. (2013/2015).

    El bucle externo (episodios) y la interacción con el entorno
    son responsabilidad del Agent; aquí solo gestionamos:
      - La red neuronal Q (online)
      - La memoria de repetición D
      - El paso de aprendizaje (gradient descent sobre el error TD)

    Parámetros:
      - state_size  : dimensión del espacio de estados (int para discreto, o shape para continuo)
      - action_size : número de acciones disponibles
      - q_network   : instancia de QNetwork (ya construida externamente)
      - alpha       : tasa de aprendizaje del optimizador
      - gamma       : factor de descuento
      - memory_size : capacidad N de la replay memory
      - batch_size  : tamaño del minibatch para cada paso de gradiente
      - min_memory  : mínimo de transiciones en D antes de empezar a entrenar
      - device      : 'cpu' o 'cuda'
    """

    def __init__(
        self,
        state_size,
        action_size,
        q_network,
        alpha=1e-3,
        gamma=0.99,
        memory_size=10_000,
        batch_size=64,
        min_memory=None,
        device=None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size

        # Si no se especifica, empezamos a entrenar cuando tengamos al menos un batch
        self.min_memory = min_memory if min_memory is not None else batch_size

        # Donde se ejecuta la QNetwork
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Red neuronal Q y optimizador
        self.q_network = q_network.to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()

        # Memoria de repetición D (capacidad N)
        self.memory = ReplayMemory(memory_size)

        # Estadísticas (compatibles con el Agent)
        self.stats = {
            "cum_training_error": 0.0,  # Suma acumulada del error TD (|y_j - Q|)
        }

        # Atributos requeridos por el Agent para compatibilidad
        # (el Agent comprueba hasattr(learner, 'w') para saber si es aproximador)
        # Como DQN es un aproximador, exponemos 'w' como los parámetros planos de la red
        self._update_w()

    # Interfaz requerida por el Agent
    def reset(self):
        """
        Reinicia los parámetros de la red y la memoria al inicio de cada run.
        También pone a cero las estadísticas acumuladas.
        """
        # Reiniciar pesos de la red con inicialización por defecto de PyTorch
        for layer in self.q_network.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        # Vaciar la memoria de experiencias
        self.memory = ReplayMemory(self.memory.memory.maxlen)

        # Resetear estadísticas
        self.stats["cum_training_error"] = 0.0

        # Actualizar el vector w (para que el Agent pueda almacenarlo)
        self._update_w()

    def start_episode(self):
        """Llamado al inicio de cada episodio. No se necesita acción especial en DQN."""
        return

    def end_episode(self):
        """Llamado al final de cada episodio. No se necesita acción especial en DQN."""
        return

    def step(self, state, action, reward, next_state, done):
        """
        Ejecuta un paso del Algoritmo 1:
          1. Almacena la transición (phi_t, a_t, r_t, phi_{t+1}) en D.
          2. Muestrea un minibatch aleatorio de D.
          3. Calcula los targets y_j.
          4. Realiza un paso de gradient descent sobre (y_j - Q(phi_j, a_j; theta))^2.

        Parámetros:
          - state      : estado actual (array numpy o similar)
          - action     : acción tomada (int)
          - reward     : recompensa recibida (float)
          - next_state : estado siguiente (array numpy o similar)
          - done       : True si el episodio ha terminado (bool)
        """
        # 1. Almacenar transición en la memoria de repetición
        self.memory.push(state, action, reward, next_state, done)

        # No entrenamos hasta tener suficientes experiencias en D
        if len(self.memory) < self.min_memory:
            return

        # 2. Muestrear minibatch aleatorio de D
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convertir a tensores
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(np.array(dones)).to(self.device)

        # 3. Calcular los targets y_j
        # Para estados terminales phi_{j+1}: y_j = r_j
        # Para estados no terminales: y_j = r_j + gamma * max_{a'} Q(phi_{j+1}, a'; theta)
        with torch.no_grad():
            # Q(phi_{j+1}, a'; theta) para todos los a'
            next_q_values = self.q_network(next_states_t)                  # [batch, action_dim]
            max_next_q    = next_q_values.max(dim=1)[0]                    # [batch]
            # Si done=True, el término futuro se anula (1 - done)
            targets = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)  # y_j

        # 4. Calcular Q(phi_j, a_j; theta) para las acciones tomadas ----
        current_q_values = self.q_network(states_t)                        # [batch, action_dim]
        q_pred = current_q_values.gather(1, actions_t).squeeze(1)          # [batch]

        # 5. Paso de gradient descent: minimizar (y_j - Q(phi_j, a_j; theta))^2
        loss = self.loss_fn(q_pred, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Logging de estadísticas
        td_error = (targets - q_pred).abs().mean().item()
        self.stats["cum_training_error"] += td_error

        # Actualizar vector w plano (para que el Agent pueda almacenarlo al final del run)
        self._update_w()

    def q_values(self, state):
        """
        Devuelve los valores Q para un estado dado usando la red neuronal.
        Compatible con el Agent, que llama a learner.q_values(state) para aproximadores.

        Parámetro:
          - state : estado actual (array numpy)
        Retorna:
          - numpy array de forma [action_size] con los valores Q
        """
        state_t = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.q_network(state_t).squeeze(0).cpu().numpy()
        return q_vals

    # Métodos auxiliares
    def _update_w(self):
        """
        Aplana todos los parámetros de la red en un único vector numpy 'w'.
        El Agent usa hasattr(learner, 'w') para detectar aproximadores y
        almacena learner.w al final de cada run.
        """
        self.w = np.concatenate(
            [p.detach().cpu().numpy().ravel() for p in self.q_network.parameters()]
        )

    @property
    def d(self):
        """Dimensión del vector de pesos w (requerido por el Agent para inicializar qtables)."""
        return len(self.w)