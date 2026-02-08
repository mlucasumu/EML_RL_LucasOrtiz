"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import algorithms


def get_algorithm_label(algo: algorithms.Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__
    if isinstance(algo, algorithms.EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    elif isinstance(algo, algorithms.Softmax):
        label += f" (temp={algo.temp})"
    elif isinstance(algo, algorithms.PreferenceGradient):
        label += f" (alpha={algo.alpha})"
    elif isinstance(algo, UCB1):
        label += f" (c={algo.c})"
    elif isinstance(algo, UCB2):
        label += f" (alpha={algo.alpha})"
    elif isinstance(algo, UCB1Tuned):
        label += f" (rewards={algo.rewards})"
    # elif isinstance(algo, OtroAlgoritmo):
    #     label += f" (parametro={algo.parametro})"
    # Añadir más condiciones para otros algoritmos aquí
    else:
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[algorithms.Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[algorithms.Algorithm]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """

    raise NotImplementedError("Esta función aún no ha sido implementada.")


def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[algorithms.Algorithm]):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo

    :param steps: Número de pasos de tiempo.
    :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param args: Opcional. Parámetros que consideres. P.e. la cota teórica Cte * ln(T).
    """

    raise NotImplementedError("Esta función aún no ha sido implementada.")


def plot_arm_statistics( # Obtenida con ayuda de ChatGPT
    data,
    algorithms,
    n_runs,
    optimal_arm,
    kde=True,
    bins=30,
    alpha=0.4,
    figsize=(14, 4)
):
    """
    Plotea histogramas de densidad de recompensas por brazo,
    solapando los resultados de varios algoritmos e indicando
    cuántas veces se ha elegido cada brazo por algoritmo.
    """

    # ---------
    # Pasar a formato largo
    # ---------
    records = []

    for algo_idx, algo_data in enumerate(data):
        for arm, rewards in algo_data.items():
            for r in rewards:
                records.append({
                    "algorithm": f"a{algo_idx}: {get_algorithm_label(algorithms[algo_idx])}",
                    "arm": arm,
                    "reward": r
                })

    df = pd.DataFrame(records)

    # ---------
    # Contadores: veces elegido cada brazo por algoritmo
    # ---------
    counts = (
        df.groupby(["arm", "algorithm"])
          .size()
          .reset_index(name="count")
    )

    # Crear un dict: arm -> ["Alg 0: x veces", "Alg 1: y veces", ...]
    arm_labels = {}

    for arm in counts["arm"].unique():
        lines = ["Recompensa"]
        arm_counts = counts[counts["arm"] == arm]

        for _, row in arm_counts.iterrows():
            lines.append(f"{row['algorithm'].split()[0]} {row['count']/n_runs} veces/run")

        arm_labels[arm] = "\n".join(lines)

    # ---------
    # Plot
    # ---------
    sns.set_theme(style="whitegrid")

    g = sns.FacetGrid(
        df,
        col="arm",
        hue="algorithm",
        sharex=False,
        sharey=True,
        height=figsize[1],
        aspect=figsize[0] / (figsize[1] * df["arm"].nunique())
    )

    g.map(
        sns.histplot,
        "reward",
        bins=bins,
        stat="density",
        kde=kde,
        alpha=alpha
    )

    g.add_legend(title="Algoritmo")
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, 0.95), 
        ncol=3, 
        title="Distribución de recompensas obtenidas por brazo y algoritmo", 
        title_fontproperties = {'size':16, 'weight':'bold'},
        frameon=False,
    )
    g.set_titles(col_template="Brazo {col_name}")

    # ---------
    # Títulos personalizados por brazo
    # ---------
    for ax in g.axes.flat:
        arm = ax.get_title().replace("arm = ", "").replace("Brazo ", "")
        try:
            arm = int(arm)
        except ValueError:
            pass

        if arm == optimal_arm:
            ax.set_title(f"Brazo {arm}\nÓptimo")
        else:
            ax.set_title(f"Brazo {arm}")

        ax.set_xlabel(arm_labels[arm])
    
    plt.tight_layout()
    plt.show()

