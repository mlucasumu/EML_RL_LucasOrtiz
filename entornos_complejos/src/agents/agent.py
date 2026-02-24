'''
El Agent gestiona la interacción con el entorno y la generación de episodios.
Además, recopila estadísticas durante el proceso de aprendizaje.
Es agnóstico al algoritmo de aprendizaje y a la política empleada.
'''

class Agent:

    def __init__(self, env, learner, policy):
        self.env = env
        self.learner = learner
        self.policy = policy

    def train(self, num_episodes):

        # Ejecutar un episodio
        for ep in range(num_episodes):
            
            # Inicializar entorno y learner
            state, _ = self.env.reset()
            self.learner.start_episode()

            done = False

            while not done:
                action = self.policy.select_action(state) # Seleccionar acción en base a política
                next_state, reward, terminated, truncated, _ = self.env.step(action) # Tomar acción y transitar a nuevo estado
                done = terminated or truncated

                self.learner.step(state, action, reward, next_state, done) # Pasar la nueva información al algoritmo de aprendizaje

                state = next_state

            self.learner.end_episode() # Avisar al algoritmo de aprendizaje de que el episodio ha acabado (necesario en Monte Carlo, p.ej.)

        # TODO: Devolver tabla Q y estadísticas que habrá que recopilar durante la ejecución. El wrapper RecordEpisodeStatistics de Gymnasium puede ser útil.