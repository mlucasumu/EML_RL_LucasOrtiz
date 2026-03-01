import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np


def smooth_rolling_average(arr, rolling_window=10):
    moving_average = (
        np.convolve(
            np.array(arr).flatten(), np.ones(rolling_window), mode="valid"
        )
        / rolling_window
    )
    return moving_average

def undo_cumsum(arr): # https://stackoverflow.com/a/38666977
    new_arr = list(arr)
    new_arr[1:] -= arr[:-1].copy() 
    return new_arr


def plot_optimal_path_CliffWalker(Q, algorithm_name, max_steps=100): # Obtenida con ayuda de ChatGPT
    """
    Dibuja la ruta greedy derivada de una tabla Q
    para CliffWalking-v0 (4x12).
    """
    plt.rcParams.update(plt.rcParamsDefault)

    n_rows = 4
    n_cols = 12

    start_state = 3 * n_cols + 0
    goal_state  = 3 * n_cols + 11

    def state_to_pos(state):
        return divmod(state, n_cols)

    # Política greedy
    policy = np.argmax(Q, axis=1)

    path = []
    state = start_state
    
    for _ in range(max_steps):

        action = policy[state]
        row, col = state_to_pos(state)
        path.append((row, col, action))

        # acciones Gymnasium:
        # 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
        if action == 0:
            row = max(row - 1, 0)
        elif action == 1:
            col = min(col + 1, n_cols - 1)
        elif action == 2:
            row = min(row + 1, n_rows - 1)
        elif action == 3:
            col = max(col - 1, 0)

        next_state = row * n_cols + col

        # Si cae en el cliff: termina?
        if row == 3 and 1 <= col <= 10:
            break

        if next_state == goal_state:
            break

        state = next_state

    # Construimos grid para visualizar
    grid = np.zeros((n_rows, n_cols))

    # Marcar cliff
    grid[3, 1:11] = -1

    # Marcar goal
    grid[3, 11] = 1

    # Marcar inicio
    grid[3, 0] = 2

    # Plot
    plt.figure()
    im = plt.imshow(grid)

    # Texto y camino
    for r,c,a in path:
        if a == 0:
            flecha = '↑'
        elif a == 1:
            flecha = '→'
        elif a == 2:
            flecha = '↓'
        elif a == 3:
            flecha = '←'
        plt.text(c, r, flecha)

    # Leyenda https://stackoverflow.com/a/40666180
    colors = [im.cmap(im.norm(value)) for value in [0, -1, 1, 2]]
    labels = ["Suelo", "Acantilado", "Objetivo", "Inicio"]
    patches =[mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    
    plt.title(f"Camino de la política greedy obtenida con {algorithm_name}")
    plt.show()


def plot_metric_per_episode(metric_per_learner:list[list], 
                            title:str,
                            y_label:str,
                            legend_labels:list[str], 
                            log_scale:bool=False,
                            rolling_window:int=1):
    '''
    Genera la gráfica de una métrica (recompensas, error, etc.) en función del episodio.

    :param metric_per_learner: Valores obtenidas por cada algoritmo y episodio.
    :param title: Título de la gráfica.
    :param y_label: Nombre del eje Y.
    :param legend_labels: Etiquetas para la leyenda.
    :param log_scale: Indica si se quiere usar escala logarítmica en el eje Y.
    :param rolling_window: Tamaño de la ventana para la media deslizante.
    '''
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(12, 5))
    for idx, metric in enumerate(metric_per_learner):
        metric_smoothed = smooth_rolling_average(metric, rolling_window)
        label = legend_labels[idx]
        lines = plt.plot(metric, linewidth=1.5, alpha=0.3) # Sin suavizar
        color = lines[0].get_color()
        plt.plot(metric_smoothed, label=label, linewidth=1.5, color=color) # Suavizado
    
    if log_scale:
        plt.yscale('symlog', base=2)
        plt.yticks(plt.yticks()[0], [f"{tick}" for tick in plt.yticks()[0]])

    texto_ventana = ""
    if rolling_window > 1:
        texto_ventana = f" (vent. deslizante = {rolling_window})"

    plt.xlabel('Episodios', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title + texto_ventana, fontsize=16)
    plt.legend(title='Algoritmos')
    sns.move_legend(plt.gca(), "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def plot_rewards(rewards_per_learner:list[list[float]], legend_labels:list[str], log_scale:bool=False, rolling_window:int=1):
    plot_metric_per_episode(rewards_per_learner,
                            title="Recompensas promedio por episodio",
                            y_label="Suma de las recompensas",
                            legend_labels=legend_labels,
                            log_scale=log_scale,
                            rolling_window=rolling_window
                            )
    
def plot_training_errors(errors_per_learner:list[list[float]], legend_labels:list[str], log_scale:bool=False, rolling_window:int=1):
    errors_no_acum = [undo_cumsum(errors) for errors in errors_per_learner]
    plot_metric_per_episode(errors_no_acum,
                            title="Error de entrenamiento promedio por episodio",
                            y_label="Suma de los errores",
                            legend_labels=legend_labels,
                            log_scale=log_scale,
                            rolling_window=rolling_window
                            )
    
def plot_episode_lengths(lengths_per_learner:list[list[float]], legend_labels:list[str], log_scale:bool=False, rolling_window:int=1):
    plot_metric_per_episode(lengths_per_learner,
                            title="Longitud promedio de cada episodio",
                            y_label="Nº de pasos",
                            legend_labels=legend_labels,
                            log_scale=log_scale,
                            rolling_window=rolling_window
                            )
    
def plot_cumulative_training_errors(errors_per_learner:list[list[float]], legend_labels:list[str], log_scale:bool=False, rolling_window:int=1):
    plot_metric_per_episode(errors_per_learner,
                            title="Error de entrenamiento acumulado por episodio",
                            y_label="Error acumulado",
                            legend_labels=legend_labels,
                            log_scale=log_scale,
                            rolling_window=rolling_window
                            )
    

def plot_average_metric_per_alpha(metrics_per_alpha_per_learner:dict[dict],
                                  metric_key:str,
                                  last_episodes_ratio:float,
                                  title:str,
                                  y_label:str,  
                                  log_scale:bool=False):
    
    '''
    Genera la gráfica de una métrica (recompensas, error, etc.) en función del valor de alpha.
    La métrica ploteada será la media de los últimos episodios para cada valor de alpha. 
    El número de episodios a utilizar se indica con el parámetro last_episodes_ratio.

    :param metrics_per_alpha_per_learner: Diccionario que contiene, para cada valor de alpha,
    los resultados de cada uno de los algoritmos. Su estructura debe ser:
    {
        alpha1: {
            algoritmo1: {
                'qtable': ...,
                'rewards': ...,
                'episode_lengths': ...,
                'cum_errors': ...
            }
        }
    }
    :param metric_key: Clave del diccionario interno que indica la métrica a plotear.
    :param last_episodes_ratio: Porcentaje de los últimos episodios a utilizar para calcular la media.
    :param title: Título de la gráfica.
    :param y_label: Etiquetas del eje Y.
    :param legend_labels: Etiquetas para la leyenda.
    :param log_scale: Indica si se quiere usar escala logarítmica en el eje Y.
    '''
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.figure(figsize=(10, 5))

    alphas = list(metrics_per_alpha_per_learner.keys())

    for algo in metrics_per_alpha_per_learner[alphas[0]].keys():
        algo_values = []
        for alpha in alphas:
            all_values = metrics_per_alpha_per_learner[alpha][algo][metric_key]
            num_last_episodes = int(len(all_values) * last_episodes_ratio)
            alpha_value = all_values[-num_last_episodes:].mean()
            algo_values.append(alpha_value)
        
        lines = plt.plot(alphas, algo_values, label=algo, linewidth=1.5, alpha=1, marker='o', markersize=4)

    if log_scale:
        plt.yscale('symlog', base=2)
        plt.yticks(plt.yticks()[0], [f"{tick}" for tick in plt.yticks()[0]])

    plt.xlabel('Valor de alpha', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(f'{title} en los últimos {num_last_episodes} episodios', fontsize=16)
    plt.legend(title='Algoritmos')
    sns.move_legend(plt.gca(), "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

def plot_average_reward_per_alpha(metrics_per_alpha_per_learner:dict[dict], last_episodes_ratio:float, log_scale:bool):
    plot_average_metric_per_alpha(metrics_per_alpha_per_learner,
                                  metric_key='rewards',
                                  last_episodes_ratio=last_episodes_ratio,
                                  title='Promedio de recompensas',
                                  y_label='Recompensa promedio',
                                  log_scale=log_scale)
    
def plot_average_episode_length_per_alpha(metrics_per_alpha_per_learner:dict[dict], last_episodes_ratio:float, log_scale:bool):
    plot_average_metric_per_alpha(metrics_per_alpha_per_learner,
                                  metric_key='episode_lengths',
                                  last_episodes_ratio=last_episodes_ratio,
                                  title='Promedio de longitudes',
                                  y_label='Longitud promedio',
                                  log_scale=log_scale)