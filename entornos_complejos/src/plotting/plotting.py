import matplotlib.pyplot as plt
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
    plot_metric_per_episode(errors_per_learner,
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