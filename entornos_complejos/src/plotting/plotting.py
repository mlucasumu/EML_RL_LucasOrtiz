import matplotlib.pyplot as plt
import seaborn as sns


def plot_metric_per_episode(metric_per_learner:list[list], 
                            title:str,
                            y_label:str,
                            legend_labels:list[str], 
                            log_scale:bool):
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
    for idx, rewards in enumerate(metric_per_learner):
        label = legend_labels[idx]
        plt.plot(rewards, label=label, linewidth=1.5)
    
    if log_scale:
        plt.yscale('symlog', base=2)
        plt.yticks(plt.yticks()[0], [f"{tick}" for tick in plt.yticks()[0]])

    plt.xlabel('Episodios', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(title='Algoritmos')
    sns.move_legend(plt.gca(), "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def plot_rewards(rewards_per_learner:list[list[float]], legend_labels:list[str], log_scale:bool):
    plot_metric_per_episode(rewards_per_learner,
                            title="Recompensas promedio por episodio",
                            y_label="Suma de las recompensas",
                            legend_labels=legend_labels,
                            log_scale=log_scale
                            )
    
def plot_training_errors(errors_per_learner:list[list[float]], legend_labels:list[str], log_scale:bool):
    plot_metric_per_episode(errors_per_learner,
                            title="Error de entrenamiento promedio por episodio",
                            y_label="Suma de los errores",
                            legend_labels=legend_labels,
                            log_scale=log_scale
                            )
    
def plot_episode_lengths(lengths_per_learner:list[list[float]], legend_labels:list[str], log_scale:bool):
    plot_metric_per_episode(lengths_per_learner,
                            title="Longitud promedio de cada episodio",
                            y_label="Nº de pasos",
                            legend_labels=legend_labels,
                            log_scale=log_scale
                            )