# Trabajos de Aprendizaje por Refuerzo

## Información

- **Alumnos:** Lucas Marín, Marta; Lucas Robles, Francisco José; Ruiz Ortiz, Jorge
- **Asignatura:** Extensiones de Machine Learning
- **Curso:** 2025/2026
- **Grupo:** LucasOrtiz

## Descripción

En este repositorio se implementan y evalúan una gran variedad de algoritmos del ámbito del Aprendizaje por Refuerzo (AR), divididos en dos familias principales:

-  Por un lado, se ha estudiado el clásico problema del **bandido de los k brazos**, para lo cual se han desarrollado algoritmos de tres clases: $\epsilon$-greedy, ascenso de gradiente y métodos UCB. 

- Por otra parte, se han evaluado **problemas más complejos** que hemos resuelto mediante métodos tabulares, si el espacio de observaciones es finito, o mediante métodos aproximados, si el espacio de estados es infinito o excesivamente grande como para contenerlo en una matriz. En cuanto al primer tipo de métodos, se han estudiado dos clases de algoritmos: los métodos de Monte Carlo y los de Diferencias Temporales, cada uno con sus múltiples versiones. Con respecto a los métodos aproximados, se han analizado los algoritmos de SARSA semi-gradiente y Deep Q-Learning.

## Estructura

El repositorio se divide en dos directorios:

- En la carpeta ``./k_brazos`` se encuentran los tres cuadernos que recopilan el análisis de cada una de las tres clases de métodos de resolución del problema del bandido de k brazos. Dentro de ``./k_brazos/src`` se implementan diversos módulos con el código python necesario para llevar a cabo los experimentos de esta parte.

- En la carpeta ``./entornos_complejos`` se sitúan dos notebooks que analizan varios algoritmos de la familia de los métodos tabulares y de los métodos aproximados. Del mismo modo que en el problema del bandido, el directorio ``./entornos_complejos/src`` recopila los ficheros .py requeridos para ejecutar los experimentos. Además, dentro de ``./entornos_complejos/examples`` incluimos algunos notebooks facilitados por el profesor de la asignatura.

## Instalación y Uso

Si se quiere ejecutar los experimentos en **Google Colab**, es preciso abrir el fichero ``./main.ipynb``, que incorpora un enlace a Colab en la parte superior. Tras pulsar en dicho enlace, se abrirá el notebook en Google Colab, tras lo cual se podrá navegar al resto de notebooks del repositorio mediante los enlaces que se encuentran en el propio cuaderno (una vez lo hayamos ejecutado).

## Tecnologías Utilizadas

Para la elaboración de la parte de aprendizaje en entornos complejos nos hemos apoyado en las funcionalidades de la librería Gymnasium, que proporciona una API sencilla y abstracta para crear el bucle de aprendizaje por refuerzo de agentes inteligentes. Además, utilizamos la librería PyTorch para implementar las redes neuronales necesarias en los algoritmos de la familia Deep Q-Learning.