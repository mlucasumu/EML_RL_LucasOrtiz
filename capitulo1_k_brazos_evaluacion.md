## 1.4. Evaluación y Experimentos

### 1.4.1. Configuración Experimental
Para garantizar la validez científica y la total reproducibilidad de los resultados a lo largo del estudio de las tres familias de distribuciones (Normal, Binomial y Bernoulli), el entorno de pruebas fue estandarizado con los siguientes hiperparámetros de simulación global:
- **Ejecuciones (*Runs*):** $N = 500$ pruebas independientes estocásticas. Las curvas finales presentadas corresponden a la matriz promediada probabilísticamente sobre este número total de semillas.
- **Horizonte de Simulador (*Steps*):** $T = 300$ secuencias temporales de decisión por ejecución.
- **Semilla Base Estática:** Se inyectó globalmente una semilla matemática (aleatoria constante, habitualmente `np.random.seed(1024)`) para inicializar la topología generatriz del bandido de forma idéntica ante la comparativa de los tres algoritmos.

Para llevar a cabo las simulaciones, se definieron los siguientes espectros de hiperparámetros iniciales para cada modelo inteligente (Tabla \ref{tab:Parametros}):

\begin{table*}[t]
\centering
\caption{Hiperparámetros definidos y comparados por familia algorítmica.}
\label{tab:Parametros}
\begin{tabular}{|l|l|l|}
\hline
\textbf{Familia Algorítmica} & \textbf{Configuración / Algoritmo} & \textbf{Hiperparámetros Evaluados} \\ \hline
Exploración-Explotación & $\epsilon$-Greedy & $\epsilon \in \{0.0, 0.01, 0.1\}$ \\ \hline
Exploración-Explotación & $\epsilon$-Decay & Tasa de decaimiento $\lambda \in \{0.8, 0.99, 0.9999\}$ \\ \hline
Ascenso de Gradiente & Softmax & Temperatura $\tau \in \{0.5, 1.0, 2.0\}$ \\ \hline
Ascenso de Gradiente & Gradiente de Preferencias & Tasa de aprendizaje $\alpha \in \{0.5, 1.0, 2.0\}$ \\ \hline
Límite Sup. Condianza & UCB1 & Constante de exploración $c \in \{0.5, 1.0, 1.5\}$ \\ \hline
Límite Sup. Condianza & UCB2 & Expansión de confianza temporal $\alpha \in \{0.25, 0.5, 0.75\}$ \\ \hline
Límite Sup. Condianza & UCB1-Tuned & Ninguno (Varianza de la Recompensa empírica) \\ \hline
\end{tabular}
\end{table*}

### 1.4.2. Métricas de Desempeño
La evaluación del rendimiento integral de los agentes propuestos se analizó multidimensionalmente observando su desempeño iterativo bajo 4 prismas métricos:
1. **Recompensa Promedio Continua (*Average Reward*):** Traza iterativa de la estimación $\bar{R_t}$; refleja la capacidad del agente de percibir y adaptar ganancia bruta promedio.
2. **Ratio Probabilístico de Selección (*% Optimal Action*):** Discrimina de forma binaria el acierto o el fallo explícito al localizar matemáticamente la mejor máquina subyacente $\arg\max q_*(a)$.
3. **Arrepentimiento Acumulado (*Cumulative Regret*):** Mide la acumulación de pérdida teórica u "oportunidad perdida" a nivel temporal de decantarse por opciones penalizadoras frente a un escenario "clarividente". Se busca priorizar gráficas cóncavas y rápidamente estabilizadas (planas).
4. **Histogramas Distribucionales:** Estadísticos puros del muestreo frecuentista con el comportamiento latente final del ecosistema (*Arm Statistics*).

### 1.4.3. Análisis de Resultados (Gráficas y Comparativas)

La convergencia de rendimiento obtenida tras los experimentos numéricos extraídos de \texttt{epsilon\_greedy.ipynb}, \texttt{ascenso\_gradiente.ipynb} y \texttt{UCB\_algorithms.ipynb} ha posibilitado la categorización de parámetros óptimos para las tres ramas del estado del arte, englobados en la siguiente matriz global (Tabla \ref{tab:Resultados}):

\begin{table*}[t]
\centering
\caption{Parámetros con mayor rendimiento cruzados por distribución.}
\label{tab:Resultados}
\begin{tabular}{|l|l|p{7cm}|}
\hline
\textbf{Familia} & \textbf{Mejor Hiperparámetro} & \textbf{Análisis Crítico} \\ \hline
$\epsilon$-Greedy   & $\epsilon = 0.1$                                  & Globalmente superior a versiones estáticas conservadoras ($\epsilon=0.01$). \\ \hline
Decaimento      & $\lambda = 0.99$                                  & Combina convergencia exponencial de exploración. Cae veloz frente a entornos como Bernoulli si su suelo ($\epsilon_{min}$) es severo.\\ \hline
Softmax         & Temperatura $\tau \in [0.5, 1.0]$                 & Curva modesta en continuas; catastrófico (20\% acierto) en dominios Bernoulli. \\ \hline
P. Gradiente    & Ratio de Aprendizaje $\alpha \in [0.5, 1.0]$      & La revelación. Su matriz de distancias absolutas mitiga por completo entornos acotados y discretos (Bernoulli).  \\ \hline
UCB             & $c = 1.0$ (o $c=0.5$ Bernoulli)                    & Curvas de convergencia teórica muy seguras y logarítmicas. Rinde maravillosamente para incertidumbres Gausianas. \\ \hline
UCB1-Tuned      & Ninguno (No paramétrico)                          & Defensa inquebrantable en todas las distribuciones; el entorno escala su nivel de exploración por la varianza natamente. \\ \hline
\end{tabular}
\end{table*}

### Resultados Familia $\epsilon$-Greedy
- **Decaimiento ($\epsilon$-Decay):** Aunque en teoría debería ser mejor, en la práctica el decaimiento no ha supuesto ninguna ventaja clara en los entornos *Normal* y *Binomial*. De hecho, ha funcionado igual o ligeramente **peor** que el modelo básico ($\epsilon$-Greedy puro). De todos los ajustes probados, reducir la probabilidad de explorar progresivamente con un multiplicador $\lambda=0.99$ fue lo más equilibrado. En cambio, si la exploración se reduce muy despacio ($\lambda=0.9999$), el algoritmo pierde demasiado tiempo escogiendo opciones al azar, acumulando muchas pérdidas.
- **Bernoulli**: En este caso particular, hemos experimentado un problema grave: si frenamos la caída de exploración demasiado pronto (fijando un mínimo inamovible de $\epsilon_{min}=0.001$), el algoritmo deja de aprender de golpe. Esto provoca que se estanque prematuramente y no logre igualar la enorme tasa de aciertos que sí consiguen otros competidores.

### Resultados Familia de Ascenso de Gradiente
- **Softmax frente a los entornos:** Mientras que el método Softmax se defiende de forma aceptable en entornos donde los premios varían de forma continua (distribución *Normal*), fracasa completamente cuando los premios son de "todo o nada" (*Bernoulli*). En este último caso, el algoritmo se atasca rápidamente, eligiendo la mejor opción apenas un 20% de las veces y acumulando muchísimas pérdidas a lo largo del tiempo.
- **Gradiente de Preferencias vs Softmax:** Este ha sido uno de los grandes descubrimientos de las pruebas. El *Gradiente de Preferencias*, al guiarse por "cuánto le gusta" una opción (porcentajes de preferencia) en lugar de intentar memorizar valores exactos de puntos, **ha superado con mucha claridad a Softmax en todos los escenarios**. Su éxito es especialmente espectacular en el entorno *Bernoulli*, donde consigue elegir la mejor palanca casi el 100% de las veces desde muy temprano. Esto hace que sus pérdidas (arrepentimiento) se queden en unos ridículos 3 puntos tras 300 intentos (frente a los 70 puntos perdidos por Softmax).

### Resultados Familia UCB
- **UCB1 vs. UCB2:** El modelo clásico UCB1 aprende bastante bien en casi todas las situaciones probadas, sobre todo cuando su nivel de confianza inicial está fijado en $c=1.0$ (para los modelos *Normal* y *Binomial*). Sin embargo, en el juego de premios binarios (*Bernoulli*) le va mejor si arranca más precavido con un $c=0.5$. Su hermano mayor, **UCB2**, muestra dos caras opuestas: **falla estrepitosamente en los entornos de Binomial y Bernoulli**, donde se atasca eligiendo constantemente la peor opción sin aprender casi nada, pero a cambio resulta ser rapidísimo y el mejor de todos cuando los premios son más naturales y variados (entornos *Normales*).
- **UCB1-Tuned**: Esta versión con "piloto automático" consiguió igualar algunos de los buenos resultados de UCB1 sin necesidad de configurar ningún parámetro a mano. Aún así, sigue pasándolo bastante mal cuando las reglas del juego son drásticas (como en el entorno *Bernoulli*), donde necesita atascarse al menos 100 intentos seguidos antes de lograr darse cuenta de cuál es el brazo ganador.



Para ver todo esto de forma mucho más clara, las siguientes gráficas (sacadas de probar los algoritmos en entornos de premios de tipo *Normal*) muestran un resumen visual perfecto. Tal y como descubrimos al analizar a fondo las simulaciones, **la gráfica del Arrepentimiento Acumulado resulta ser siempre la más útil y fácil de entender**. Esto se debe a que, a diferencia de otras medidas, en el Arrepentimiento se puede distinguir a simple vista si un algoritmo sigue aprendiendo y reduciendo sus pérdidas o si ya se ha estancado por completo.

\begin{figure}[h]
    \centering
    \includegraphics[width=\columnwidth]{k_brazos/img/eg_regret_normal.png}
    \caption{Curva de Arrepentimiento Acumulado para la familia $\epsilon$-Greedy.}
    \label{fig:egreedy_perf}
\end{figure}

En la Figura \ref{fig:egreedy_perf}, se discierne claramente cómo una caída suave del decaimiento temporal ($\lambda = 0.99$) permite achatar asintóticamente el *regret* mitigando estancamientos lineales que sí sufrían ramas no decadentes.

\begin{figure}[h]
    \centering
    \includegraphics[width=\columnwidth]{k_brazos/img/ag_regret_normal.png}
    \caption{Curva de Arrepentimiento Acumulado comparando Preferencias vs Softmax.}
    \label{fig:ag_perf}
\end{figure}

Por otro lado, la comparación que vemos en la Figura \ref{fig:ag_perf} nos demuestra algo muy interesante: aunque el algoritmo Softmax arranca bien y consigue ganancias rápidamente, basar las decisiones puramente en las **Preferencias** de cada opción frente al resto (la curva verde) resulta ser una estrategia absolutamente imbatible a medio y largo plazo siempre que se tenga configurada la tasa de aprendizaje correcta ($\alpha=1$).

\begin{figure}[h]
    \centering
    \includegraphics[width=\columnwidth]{k_brazos/img/ucb_regret_normal.png}
    \caption{Curva de Arrepentimiento Acumulado demostrando el inquebrantable éxito de UCB1.}
    \label{fig:ucb_perf}
\end{figure}

Por último, la Figura \ref{fig:ucb_perf} demuestra en la práctica lo bien que funciona la idea principal del método UCB: si castigamos fuertemente a las opciones que ya hemos probado demasiadas veces, conseguimos que el algoritmo frene en seco sus pérdidas (arrepentimiento). Todo esto lo logra con una precisión asombrosa, de forma rápida y sin necesidad de fórmulas ni ajustes complicados, tal y como atestigua el excelente recorrido de la variante con "piloto automático" (*UCB1-Tuned* en la curva verde).
