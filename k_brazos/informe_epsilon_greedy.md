# Informe de Desarrollo: Evolución del Algoritmo Epsilon-Greedy y Mejoras en el Framework

## 1. Introducción
Este informe detalla el trabajo realizado para desarrollar y estabilizar el entorno de experimentación para algoritmos de *k-armed bandits*, con un foco específico en la evolución desde el algoritmo **Epsilon-Greedy** estándar hacia su variante con decaimiento (**Epsilon-Greedy Decay**), y las mejoras de infraestructura necesarias en los archivos Python implicados en `epsilon_greedy.ipynb`.

## 2. Evolución de los Algoritmos

### 2.1. Base Abstracta (`algorithm.py`)
Partimos de una clase base abstracta `Algorithm` que define la interfaz común (`select_arm`, `update`, `reset`) y mantiene el estado básico:
- `k`: Número de brazos.
- `counts`: Número de veces que se ha seleccionado cada brazo.
- `values`: Estimación de valor Q(a) para cada brazo.

### 2.2. Epsilon-Greedy Estándar (`epsilon_greedy.py`)
La primera implementación consistió en el algoritmo clásico:
- **Lógica**: Con probabilidad $\epsilon$ explora (selecciona al azar), y con probabilidad $1-\epsilon$ explota (selecciona el brazo con mayor Q).
- **Limitación**: Un $\epsilon$ fijo mantiene una exploración constante incluso cuando el agente ya tiene mucha certeza sobre cuál es el mejor brazo, lo que impide la convergencia total al óptimo.

### 2.3. Evolución: Epsilon-Greedy con Decaimiento (`epsilon_greedy_decay.py`)
Para superar la limitación anterior, se implementó `EpsilonGreedyDecay`.
- **Innovación**: Introducción de un `decay_rate` (tasa de decaimiento).
- **Mecánica**: En cada paso $t$, actualizamos $\epsilon_{t+1} = \max(\epsilon_{min}, \epsilon_t \times \text{decay\_rate})$.
- **Justificación**: Esto permite una alta exploración al inicio (cuando el conocimiento es bajo) y una transición gradual hacia la explotación pura (cuando las estimaciones de Q son precisas), mejorando el *regret* acumulado a largo plazo.

## 3. Evolución de la Infraestructura y Archivos Auxiliares

Para que el notebook `epsilon_greedy.ipynb` funcionara correctamente y ofreciera resultados significativos, fue necesario realizar cambios estructurales importantes en los archivos fuente (`.py`).

### 3.1. Gestión de Paquetes (`algorithms/__init__.py` y `algorithms/preference_gradient.py`)
**Situación:** Se detectaron errores de importación (`ModuleNotFoundError`) debido a archivos faltantes referenciados en el paquete.
**Acción:** 
- Se implementó desde cero el módulo `preference_gradient.py` (Gradient Bandit) para completar la suite de algoritmos.
- Se limpió `__init__.py` eliminando referencias a algoritmos no implementados (`UCB1Tuned`) que causaban roturas.
**Justificación:** Garantizar que la importación del paquete `algorithms` sea robusta y no falle, permitiendo la ejecución fluida del notebook.

### 3.2. Visualización y Métricas (`plotting/plotting.py` y `plotting/__init__.py`)
**Situación:** El notebook intentaba importar `plot_regret`, pero esta función no existía o no estaba expuesta. Además, `plot_optimal_selections` estaba vacía.
**Acción:** 
- Se implementó `plot_regret`: Calcula y grafica el *regret* acumulado (diferencia entre el valor óptimo posible y el obtenido).
- Se implementó `plot_optimal_selections`: Grafica el % de veces que se eligió el mejor brazo.
- Se exportaron ambas funciones en `__init__.py`.
**Justificación:** El análisis de `Epsilon-Greedy` queda incompleto si solo miramos la recompensa promedio. El *regret* acumulado es la métrica teórica clave para comparar estrategias de exploración (constante vs. decaimiento).

### 3.3. Motor de Experimentación (`main.py`)
**Situación:** Se produjo un error `ValueError: not enough values to unpack` porque el código esperaba 3 valores de retorno (rewards, optimal, regret), pero `run_experiment` solo devolvía 2.
**Acción:** Se actualizó la función `run_experiment` para:
1. Calcular el *regret* en cada paso: $Regret_t = Q(a^*) - Q(a_t)$.
2. Acumular y devolver este tercer array de datos.
**Justificación:** Alinear la función principal con los requisitos de análisis del notebook, permitiendo estudiar no solo "cuánto ganamos" sino "cuánto dejamos de ganar".

## 4. Conclusión

La evolución propuesta ha transformado un conjunto de scripts básicos en un framework de experimentación optimizado. La inclusión de `EpsilonGreedyDecay` demuestra cómo una simple modificación dinámica en los hiperparámetros puede mejorar el rendimiento. Las correcciones en `main.py` y `plotting.py` fueron críticas para permitir la validación empírica de estas mejoras teóricas a través de visualizaciones completas en `epsilon_greedy.ipynb`.
