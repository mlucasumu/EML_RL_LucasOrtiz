import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

# from .q_network import QNetwork
from .base_learner import BaseLearner
from .replay_memory import ReplayMemory


class DoubleDQNLearner(BaseLearner):
    """
    Learner que implementa Double Deep Q-Learning (van Hasselt et al., 2015).

    La diferencia clave respecto a DQN estándar está en el cálculo del target:

      DQN:         y = r + gamma * max_{a'} Q(s', a'; w-)
      Double DQN:  y = r + gamma * Q(s', argmax_{a'} Q(s', a'; w), w-)
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^
                                            La red ONLINE selecciona la acción
                                            La red TARGET evalúa su valor

    Esto desacopla la selección de la evaluación, eliminando el sesgo de
    sobreestimación que introduce el operador max en DQN estándar.

    Parámetros:
      - state_size       : dimensión del espacio de estados
      - action_size      : número de acciones disponibles
      - q_network        : instancia de QNetwork (red online, ya construida)
      - alpha            : tasa de aprendizaje del optimizador
      - gamma            : factor de descuento
      - memory_size      : capacidad N de la replay memory
      - batch_size       : tamaño del minibatch para cada paso de gradiente
      - min_memory       : mínimo de transiciones en D antes de entrenar
      - target_update_freq : cada cuántos pasos se sincronizan los pesos
                             de la red target con los de la red online
      - device           : 'cpu' o 'cuda'
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
        target_update_freq=100,
        device=None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.min_memory = min_memory if min_memory is not None else batch_size

        # Donde se ejecuta la QNetwork
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Red ONLINE: q(s, a; w)
        # Selecciona la mejor acción: argmax_{a'} Q(s', a'; w)
        # Sus pesos se actualizan en cada paso de gradiente.
        self.online_network = q_network.to(self.device)

        # Red TARGET: q(s, a; w-)
        # Evalúa el valor de la acción seleccionada por la red online.
        # Sus pesos se actualizan periódicamente (cada target_update_freq pasos),
        # copiando los pesos de la red online. Esto estabiliza los targets y_j.
        self.target_network = copy.deepcopy(q_network).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        # La red target nunca se entrena directamente: solo recibe copias de la online
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.online_network.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()

        # Memoria de repetición D
        self.memory = ReplayMemory(memory_size)

        # Contador global de pasos (para saber cuándo actualizar la target)
        self._step_count = 0

        # Estadísticas (compatibles con el Agent)
        self.stats = {
            "cum_training_error": 0.0,
        }

        # Vector w plano de parámetros (para compatibilidad con el Agent,
        # que detecta aproximadores comprobando hasattr(learner, 'w'))
        self._update_w()

    def reset(self):
        """
        Reinicia pesos de ambas redes y la memoria al inicio de cada run.
        """
        # Reiniciar pesos de la red online
        for layer in self.online_network.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        # Sincronizar la target con la online recién reiniciada
        self.target_network.load_state_dict(self.online_network.state_dict())

        # Vaciar memoria de experiencias
        self.memory = ReplayMemory(self.memory.memory.maxlen)

        # Resetear contadores y estadísticas
        self._step_count = 0
        self.stats["cum_training_error"] = 0.0

        self._update_w()

    def start_episode(self):
        return

    def end_episode(self):
        return

    def step(self, state, action, reward, next_state, done):
        """
        Ejecuta un paso de Double DQN:

          1. Almacena (phi_t, a_t, r_t, phi_{t+1}) en D.
          2. Muestrea un minibatch aleatorio de D.
          3. Calcula los targets y_j con la fórmula Double DQN:
               a* = argmax_{a'} Q(phi_{j+1}, a'; w)       <- red ONLINE selecciona
               y_j = r_j + gamma * Q(phi_{j+1}, a*; w-)  <- red TARGET evalúa
          4. Gradient descent sobre (y_j - Q(phi_j, a_j; w))^2.
          5. Cada target_update_freq pasos, copia w -> w-.

        Parámetros:
          - state      : estado actual
          - action     : acción tomada (int)
          - reward     : recompensa recibida (float)
          - next_state : estado siguiente
          - done       : True si el episodio terminó
        """
        # 1. Almacenar transición en D
        self.memory.push(state, action, reward, next_state, done)
        self._step_count += 1

        # No entrenamos hasta tener suficientes experiencias
        if len(self.memory) < self.min_memory:
            return

        # 2. Muestrear minibatch de D
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t      = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t     = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards_t     = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t       = torch.FloatTensor(np.array(dones)).to(self.device)

        # 3. Calcular targets y_j con Double DQN
        with torch.no_grad():
            # Paso A: la red ONLINE selecciona la mejor acción en s'
            # a* = argmax_{a'} Q(phi_{j+1}, a'; w)
            online_next_q   = self.online_network(next_states_t)   # [batch, action_dim]
            best_actions    = online_next_q.argmax(dim=1, keepdim=True)  # [batch, 1]

            # Paso B: la red TARGET evalúa el valor de esa acción
            # Q(phi_{j+1}, a*; w-)  →  desacopla selección y evaluación
            target_next_q   = self.target_network(next_states_t)   # [batch, action_dim]
            target_best_q   = target_next_q.gather(1, best_actions).squeeze(1)  # [batch]

            # y_j = r_j  (terminal)  ó  r_j + gamma * Q(s', a*; w-)  (no terminal)
            targets = rewards_t + self.gamma * target_best_q * (1.0 - dones_t)

        # 4. Q(phi_j, a_j; w) para las acciones realmente tomadas
        current_q  = self.online_network(states_t)                  # [batch, action_dim]
        q_pred     = current_q.gather(1, actions_t).squeeze(1)      # [batch]

        # 5. Paso de gradient descent
        loss = self.loss_fn(q_pred, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Logging de estadísticas
        td_error = (targets - q_pred).abs().mean().item()
        self.stats["cum_training_error"] += td_error

        # 6. Actualizar la red TARGET periódicamente
        # Cada target_update_freq pasos copiamos w -> w-
        if self._step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

        # Actualizar vector w plano para el Agent
        self._update_w()

    def q_values(self, state):
        """
        Devuelve los valores Q de la red ONLINE para un estado dado.
        La red online es la que se usa para seleccionar acciones (política actual).

        Parámetro:
          - state : estado actual (array numpy)
        Retorna:
          - numpy array [action_size] con los valores Q
        """
        state_t = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.online_network(state_t).squeeze(0).cpu().numpy()
        return q_vals

    def _update_w(self):
        """
        Aplana los parámetros de la red ONLINE en un vector numpy 'w'.
        El Agent usa hasattr(learner, 'w') para detectar aproximadores.
        """
        self.w = np.concatenate(
            [p.detach().cpu().numpy().ravel() for p in self.online_network.parameters()]
        )

    @property
    def d(self):
        """Dimensión del vector de pesos w (requerido por el Agent)."""
        return len(self.w)