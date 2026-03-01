import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from .base_learner import BaseLearner

class QNetwork(nn.Module):
    """Red Neuronal simple para aproximar la función Q en DQN"""
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Memoria de Experiencia (Experience Replay) para descorrelacionar las muestras"""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
        
    def __len__(self):
        return len(self.memory)

class DQN(BaseLearner):
    """
    Agente Deep Q-Learning usando PyTorch.
    Incluye Replay Buffer y Target Network para estabilizar el aprendizaje.
    """
    def __init__(self, state_size, action_size, learning_rate, gamma, 
                 buffer_capacity=10000, batch_size=64, target_update_freq=10):
        super().__init__(state_size, action_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Red Principal (se entrena en cada paso)
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        # Red Objetivo (se actualiza esporádicamente para dar estabilidad a los targets de Bellman)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.memory = ReplayBuffer(buffer_capacity)
        self.steps_done = 0
        
        self.reset()

    def start_episode(self):
        return

    def get_q_values(self, state):
        """Predice los valores Q para un estado dado usando la red principal"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.cpu().numpy()[0]

    def step(self, state, action, reward, next_state, done):
        """Almacena la experiencia y ejecuta un paso de optimización de la red si hay suficientes datos"""
        # 1. Guardar en memoria
        self.memory.push(state, action, reward, next_state, done)
        self.steps_done += 1
        
        # 2. Replay Experience y Optimización
        if len(self.memory) >= self.batch_size:
            self._optimize_model()
            
        # 3. Actualizar la red objetivo periódicamente
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def _optimize_model(self):
        """Ejecuta un paso del SGD (Adam) calculando la pérdida de Bellman en un mini-batch"""
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convertir a Tensores de PyTorch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Calcular Q(s_t, a) usando la red principal
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Calcular max_a Q(s_{t+1}, a) usando la TARGET Network
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
        # El target Y_t de DQN: Recompensa inmediata + Recompensa futura descontada (0 si ha terminado)
        expected_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        # Pérdida MSE Loss entre lo predicho y el Target de Bellman
        loss = self.criterion(current_q_values, expected_q_values)
        
        # Optimizar el modelo
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping para evitar explosión de gradientes (común en DQN)
        for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        self.stats['cum_training_error'] += loss.item()

    def end_episode(self):
        return

    def reset(self):
        self.qtable = None # Retrocompatibilidad
        self.stats = {'cum_training_error': 0.0}
