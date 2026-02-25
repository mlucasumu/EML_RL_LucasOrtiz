'''
El Agent gestiona la interacción con el entorno y la generación de episodios.
Además, recopila estadísticas durante el proceso de aprendizaje.
Es agnóstico al algoritmo de aprendizaje y a la política empleada.
'''

import numpy as np
from tqdm import tqdm

class Agent:

    def __init__(self, env, learner, policy):
        self.env = env
        self.learner = learner
        self.policy = policy

    def train(self, num_episodes, n_runs, seed=42):

        # Estadísticas
        rewards = np.zeros(num_episodes)
        episode_lengths = np.zeros(num_episodes)
        qtables = np.zeros((n_runs, self.learner.state_size, self.learner.action_size))
        learner_stats = {k: np.zeros(num_episodes) for k,v in self.learner.stats.items()}

        np.random.seed(seed=seed)

        for run in tqdm(range(n_runs)): 
            self.learner.reset() # Reseteamos la tabla Q en cada ejecución
            
            for ep in range(num_episodes):
                
                # Inicializar entorno y learner
                state, _ = self.env.reset(seed=seed)
                self.learner.start_episode()

                done = False

                while not done:
                    action = self.policy.select_action(state, self.learner.qtable) # Seleccionar acción en base a política
                    next_state, reward, terminated, truncated, _ = self.env.step(action) # Tomar acción y transitar a nuevo estado
                    done = terminated or truncated

                    self.learner.step(state, action, reward, next_state, done) # Pasar la nueva información al algoritmo de aprendizaje
                    state = next_state

                    # Logging
                    rewards[ep] += reward
                    episode_lengths[ep] += 1

                self.learner.end_episode() # Avisar al algoritmo de aprendizaje de que el episodio ha acabado (necesario en Monte Carlo, p.ej.)
                for stat in learner_stats.keys():
                    learner_stats[stat][ep] += self.learner.stats[stat]
            
            qtables[run] = self.learner.qtable

        # Media de los resultados entre todas las runs
        rewards /= n_runs
        episode_lengths /= n_runs
        learner_stats = {k: v/n_runs for k,v in learner_stats.items()}
        qtable = np.mean(qtables, axis=0)

        return qtable, rewards, episode_lengths, learner_stats