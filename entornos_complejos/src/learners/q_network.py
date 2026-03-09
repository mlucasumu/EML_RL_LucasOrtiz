import torch.nn as nn  # Módulo para definir modelos de redes neuronales.
import torch.nn.functional as F  # Funciones de activación y utilidades de PyTorch.

class QNetwork(nn.Module):
    """
    Red neuronal para aproximar la función Q.

    Parámetros:
      - state_dim (int): Dimensión del estado (para CartPole: 4).
      - action_dim (int): Número de acciones posibles (para CartPole: 2).
      - hidden_dim (int): Número de neuronas en las capas ocultas (por defecto: 64).
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        # Primera capa: de estado a capa oculta de tamaño hidden_dim.
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # Segunda capa oculta.
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Capa de salida: de hidden_dim a número de acciones.
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        Propagación hacia adelante.

        Parámetro:
          - x (Tensor): Estado de entrada con forma [batch_size, state_dim].

        Retorna:
          - Tensor: Valores Q para cada acción, con forma [batch_size, action_dim].
        """
        # Aplicar la primera capa seguida de ReLU.
        x = F.relu(self.fc1(x))
        # Aplicar la segunda capa seguida de ReLU.
        x = F.relu(self.fc2(x))
        # Capa de salida sin activación, para obtener los valores Q.
        x = self.fc3(x)
        return x