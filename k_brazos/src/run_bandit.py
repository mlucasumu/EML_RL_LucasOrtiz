import numpy as np
import matplotlib.pyplot as plt
from environment import Bandit
from agents import EpsilonGreedyAgent, DecayingEpsilonGreedyAgent, OptimisticAgent, UCBAgent, GradientBanditAgent, UCB2Agent, SoftmaxAgent, EpsilonFirstAgent
import os

def run_experiment(agent_class, run_params, bandit_params, n_runs=2000, n_steps=1000):
    """
    Ejecuta un experimento para un tipo de agente dado.
    
    Args:
        agent_class: Clase del agente a instanciar.
        run_params: Diccionario con parámetros para el agente (e.g. {'epsilon': 0.1}).
        bandit_params: Diccionario con parámetros para el entorno.
        n_runs: Número de ejecuciones independientes.
        n_steps: Pasos por ejecución.
        
    Returns:
        avg_rewards: Array de tamaño (n_steps,) con la recompensa promedio en cada paso.
        optimal_actions: Array de tamaño (n_steps,) con el % de veces que se eligió la acción óptima.
        avg_cumulative_regret: Array de tamaño (n_steps,) con el regret acumulado promedio.
    """
    rewards = np.zeros((n_runs, n_steps))
    optimal_action_counts = np.zeros((n_runs, n_steps))
    regrets = np.zeros((n_runs, n_steps)) # Almacenar regret por paso
    
    for r in range(n_runs):
        bandit = Bandit(**bandit_params)
        agent = agent_class(bandit.k, **run_params)
        
        # Obtener el valor óptimo real para calcular regret
        # q_star contiene los valores reales.
        # Regret_t = max_a(q*(a)) - q*(A_t)
        max_q = np.max(bandit.q_star)
        
        for t in range(n_steps):
            action = agent.select_action()
            reward = bandit.step(action)
            agent.update(action, reward)
            
            # Cálculo de Regret instantáneo
            regret = max_q - bandit.q_star[action]
            regrets[r, t] = regret
            
            if action == bandit.best_action:
                optimal_action_counts[r, t] = 1
            
            rewards[r, t] = reward
                
    avg_rewards = rewards.mean(axis=0)
    avg_optimal_actions = optimal_action_counts.mean(axis=0)
    
    # Cumulative Regret promedio
    cumulative_regrets = np.cumsum(regrets, axis=1)
    avg_cumulative_regret = cumulative_regrets.mean(axis=0)
    
    return avg_rewards, avg_optimal_actions, avg_cumulative_regret

def plot_results(results, metric, title, filename):
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        plt.plot(data, label=name)
    plt.xlabel("Pasos")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # Configuración global
    SEED = 42
    np.random.seed(SEED)
    print(f"Semilla fijada en: {SEED}")
    
    N_RUNS = 2000 
    N_STEPS = 1000
    K = 10
    
    # Crear directorio para gráficas
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Sintetizar mejores parámetros encontrados (hardcoded de ejecución anterior)
    best_params = {
        'softmax_tau': 0.1, 
        'ucb_c': 2,
        'gradient_alpha': 0.1
    }
    
    # Ejecución General con los Mejores Parámetros para todas las distribuciones
    print("\n--- Ejecutando Comparativa Final con Mejores Parámetros (Incluyendo Regret) ---")
    
    distributions = ['normal', 'bernoulli', 'binomial']
    
    for dist in distributions:
        print(f"\nDistribución: {dist.upper()}")
        bandit_params = {'k': K, 'mu': 0, 'sigma': 1, 'stationary': True, 'distribution': dist, 'n_binomial': 5}
        
        results_reward = {}
        results_optimal = {}
        results_regret = {}
        
        # Epsilon-Greedy
        r, o, reg = run_experiment(EpsilonGreedyAgent, {'epsilon': 0.1}, bandit_params, N_RUNS, N_STEPS)
        results_reward["e-greedy (0.1)"] = r
        results_optimal["e-greedy (0.1)"] = o
        results_regret["e-greedy (0.1)"] = reg
        
        # Epsilon-First (NUEVO)
        r, o, reg = run_experiment(EpsilonFirstAgent, {'epsilon': 0.1, 'horizon': N_STEPS}, bandit_params, N_RUNS, N_STEPS)
        results_reward["e-first (0.1)"] = r
        results_optimal["e-first (0.1)"] = o
        results_regret["e-first (0.1)"] = reg

        # UCB1
        c = best_params['ucb_c']
        r, o, reg = run_experiment(UCBAgent, {'c': c}, bandit_params, N_RUNS, N_STEPS)
        results_reward[f"UCB1 (c={c})"] = r
        results_optimal[f"UCB1 (c={c})"] = o
        results_regret[f"UCB1 (c={c})"] = reg
        
        # UCB2
        r, o, reg = run_experiment(UCB2Agent, {'alpha': 0.1}, bandit_params, N_RUNS, N_STEPS)
        results_reward["UCB2 (alpha=0.1)"] = r
        results_optimal["UCB2 (alpha=0.1)"] = o
        results_regret["UCB2 (alpha=0.1)"] = reg

        # Softmax
        tau = best_params['softmax_tau']
        r, o, reg = run_experiment(SoftmaxAgent, {'tau': tau}, bandit_params, N_RUNS, N_STEPS)
        results_reward[f"Softmax (tau={tau})"] = r
        results_optimal[f"Softmax (tau={tau})"] = o
        results_regret[f"Softmax (tau={tau})"] = reg

        # Gradient
        r, o, reg = run_experiment(GradientBanditAgent, {'alpha': 0.1, 'baseline': True}, bandit_params, N_RUNS, N_STEPS)
        results_reward["Gradient"] = r
        results_optimal["Gradient"] = o
        results_regret["Gradient"] = reg

        plot_results(results_reward, "Recompensa Promedio", f"Comparativa Final {dist.capitalize()} (Recompensa)", f"plots/final_{dist}_reward.png")
        plot_results(results_optimal, "% Acción Óptima", f"Comparativa Final {dist.capitalize()} (% Óptimo)", f"plots/final_{dist}_optimal.png")
        plot_results(results_regret, "Regret Acumulado", f"Comparativa Final {dist.capitalize()} (Regret)", f"plots/final_{dist}_regret.png")

    print("Estudio finalizado.")

    # --- Estudio de Estadísticas por Brazo (Fixed Bandit) ---
    print("\n--- Ejecutando Análisis de Estadísticas por Brazo (Fixed Bandit) ---")
    
    def plot_arm_statistics(counts, avg_rewards, best_action, distribution_name, filename):
        """
        Genera un histograma con las estadísticas por brazo.
        Eje X: Brazos (con etiquetas de N y si es óptimo).
        Eje Y: Recompensa promedio obtenida.
        """
        k = len(counts)
        arms = np.arange(k)
        
        plt.figure(figsize=(14, 7))
        bars = plt.bar(arms, avg_rewards, color='skyblue', edgecolor='black')
        
        # Resaltar brazo óptimo
        bars[best_action].set_color('green')
        bars[best_action].set_edgecolor('black')
        bars[best_action].set_label('Óptimo')
        
        plt.xlabel('Brazos')
        plt.ylabel('Recompensa Promedio')
        plt.title(f'Estadísticas por Brazo - {distribution_name}\n(Promedio de Recompensas y Frecuencia de Selección)')
        
        # Etiquetas del eje X
        xtick_labels = []
        for i in range(k):
            label = f"Arm {i}\nN={int(counts[i])}"
            if i == best_action:
                label += "\n[OPTIMAL]"
            xtick_labels.append(label)
        
        plt.xticks(arms, xtick_labels)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def analyze_arm_statistics(agent_class, run_params, bandit_params, n_runs=1000, n_steps=1000, dist_name="Normal"):
        # Crear UN único entorno fijo para que los brazos tengan identidad
        fixed_bandit = Bandit(**bandit_params)
        
        # Estructuras para acumular estadísticas
        total_counts = np.zeros(fixed_bandit.k)
        total_rewards = np.zeros(fixed_bandit.k)
        
        for r in range(n_runs):
            # Reiniciar agente, pero usar el MISMO bandit
            agent = agent_class(fixed_bandit.k, **run_params)
            
            for t in range(n_steps):
                action = agent.select_action()
                reward = fixed_bandit.step(action)
                agent.update(action, reward)
                
                total_counts[action] += 1
                total_rewards[action] += reward
        
        # Calcular promedios
        # Evitar división por cero
        avg_rewards_per_arm = np.divide(total_rewards, total_counts, out=np.zeros_like(total_rewards), where=total_counts!=0)
        
        # Graficar
        filename = f"plots/arm_stats_{dist_name.lower()}.png"
        plot_arm_statistics(total_counts, avg_rewards_per_arm, fixed_bandit.best_action, dist_name, filename)
        print(f"Gráfica guardada: {filename}")

    # Ejecutar para Normal (usando Softmax como ejemplo representativo, o UCB)
    # Usaremos UCB1 con c=2 que funcionó bien
    dist = 'normal'
    params = {'k': K, 'mu': 0, 'sigma': 1, 'stationary': True, 'distribution': dist, 'n_binomial': 5}
    # Usamos UCB1 para ver cómo explora/explota
    analyze_arm_statistics(UCBAgent, {'c': best_params['ucb_c']}, params, N_RUNS, N_STEPS, "Normal_UCB1")
    
    # Ejecutar también para Epsilon-Greedy para comparar
    analyze_arm_statistics(EpsilonGreedyAgent, {'epsilon': 0.1}, params, N_RUNS, N_STEPS, "Normal_EpsilonGreedy")
    
    print("Análisis de brazos finalizado.")
