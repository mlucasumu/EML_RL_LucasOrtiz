## 1.3. Algoritmos

Para resolver el problema del bandido de $k$-brazos hemos implementado tres enfoques distintos: métodos que combinan explotar lo conocido con explorar al azar, algoritmos que premian probar opciones inciertas para dejar de dudar, y tácticas que eligen las acciones basándose en preferencias relativas en lugar de en ganancias directas. A continuación se justifican sus enfoques y se detalla el funcionamiento lógico de los pilares principales de cada familia algorítmica estudiada.

### Familia 1: Lógicas de Exploración-Explotación ($\epsilon$-Greedy)
**Justificación:** El algoritmo $\epsilon$-Greedy es el punto de partida más clásico en el Aprendizaje por Refuerzo. Su funcionamiento es muy sencillo y rápido de calcular: la mayor parte del tiempo (con probabilidad $1-\epsilon$) el agente elige la opción que hasta ahora le ha dado más beneficios, pero de vez en cuando (con una pequeña probabilidad $\epsilon$) elige una opción al azar para explorar. A partir de aquí nace una mejora llamada $\epsilon$-Decay, que soluciona el problema de explorar para siempre: va reduciendo poco a poco la probabilidad de explorar (multiplicándola por un factor $\lambda$), de modo que, a medida que el agente aprende, se vuelve más seguro y se centra en asegurar las ganancias.

**Pseudocódigo de $\epsilon$-Greedy Clásico:**
```latex
\begin{algorithm}
\caption{Algoritmo $\epsilon$-Greedy para K-Brazos}
\begin{algorithmic}[1]
\REQUIRE $\epsilon \in [0, 1]$, Total de acciones $K$
\STATE Inicializar $Q(a) \leftarrow 0$ para todo $a=1,\dots, K$
\STATE Inicializar $N(a) \leftarrow 0$ para todo $a=1,\dots, K$
\FOR{cada paso $t=1, 2, \dots, T$}
    \STATE $p \leftarrow$ Valor aleatorio de Distribución Uniforme(0, 1)
    \IF{$p < \epsilon$}
        \STATE $A_t \leftarrow$ Acción aleatoria uniforme en $\{1, \dots, K\}$
    \ELSE
        \STATE $A_t \leftarrow \arg\max_a Q(a)$  \COMMENT{Ruptura de empates al azar}
    \ENDIF
    \STATE Ejecutar $A_t$, observar recomensa $R_t$
    \STATE $N(A_t) \leftarrow N(A_t) + 1$
    \STATE $Q(A_t) \leftarrow Q(A_t) + \frac{1}{N(A_t)} \big[R_t - Q(A_t)\big]$
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

### Familia 2: Optimismo ante la Incertidumbre (UCB)
**Justificación:** A diferencia de explorar siempre al azar, los métodos de *Upper Confidence Bound* (UCB) son curiosos de forma mucho más inteligente. UCB calcula cuánta confianza tiene en lo que ya sabe de cada palanca. Si una opción se ha probado muy poco, su nivel de incertidumbre es alto, así que el algoritmo decide elegirla para "salir de dudas". Esto convierte a UCB en un método robusto y bueno para no perder el tiempo. En este trabajo hemos estudiado el algoritmo básico UCB1 (que usa un número $c$ para controlar de forma fija esa curiosidad), el modelo UCB2 (que optimiza el proceso probando las opciones en "ráfagas" o bloques cada vez más largos en lugar de recalcular a cada paso), y la variante avanzada UCB1-Tuned (que se ajusta a sí misma midiendo la variabilidad de los premios que recibe).

**Pseudocódigo de UCB2:**
```latex
\begin{algorithm}
\caption{Algoritmo UCB2}
\begin{algorithmic}[1]
\REQUIRE Constante exploratoria $\alpha \in (0, 1)$
\STATE Inicializar contador de épocas por brazo: para cada $a$, $r_a \leftarrow 0$
\STATE Iniciar explorando todos los brazos una vez, actualizando su recompensa promedio $Q(a)$
\FOR{paso de tiempo $t = K+1, \dots$}
    \STATE Seleccionar brazo $A_t \leftarrow \arg\max_a \left[ Q(a) + \sqrt{\frac{(1 + \alpha) \ln (e \cdot t / \tau(r_a))}{2 \tau(r_a)}} \right]$ \newline
           (donde $\tau(r) = \lceil (1+\alpha)^r \rceil$)
    \STATE Determinar longitud de la ráfaga: $\Delta \leftarrow \tau(r_{A_t} + 1) - \tau(r_{A_t})$
    \FOR{iteraciones $i = 1$ hasta $\Delta$}
        \STATE Ejecutar $A_t$, percibir recompensa $R$
        \STATE $N(A_t) \leftarrow N(A_t) + 1$
        \STATE $Q(A_t) \leftarrow Q(A_t) + \frac{1}{N(A_t)} \big[R - Q(A_t)\big]$
    \ENDFOR
    \STATE Actualizar época del brazo: $r_{A_t} \leftarrow r_{A_t} + 1$
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

### Familia 3: Ascenso de Gradiente Estocástico (Softmax y Preferencias)
**Justificación:** Si los algoritmos solo se fijan en los puntos exactos de cada opción ($Q$), pueden tener problemas para decidir si los premios son muy parecidos (por ejemplo, ganar $+85$ o $+84$). Los métodos basados en el Gradiente lo resuelven usando "Preferencias" ($H_a$). En vez de memorizar valores numéricos rígidos, simplemente aprenden qué opción prefieren en relación al resto y deciden usando porcentajes (con la Ley Exponencial Suave o Gibbs). En nuestras pruebas, esta estrategia del *Gradiente de Preferencias* trataremos de demostrar si es la que mejor se adapta a los entornos donde las recompensas son simplemente "todo o nada" (como sucede en el modelo de *Bernoulli*).

**Pseudocódigo de Base Gibbs / Gradiente:**
```latex
\begin{algorithm}
\caption{Algoritmo de Variación de Gradiente de Preferencias}
\begin{algorithmic}[1]
\REQUIRE Tasa de aprendizaje $\alpha > 0$
\STATE Inicializar preferencias $H(a) \leftarrow 0 \;\; \forall a$ 
\STATE Inicializar recompensa promedio de validación $\bar{R} \leftarrow 0$
\FOR{$t = 1, \dots, T$}
    \STATE Calcular probabilidades de la curva Softmax:\\
    $P(a) \leftarrow \frac{\exp(H(a))}{\sum_{b=1}^K \exp(H(b))} \quad \forall a$
    \STATE Elegir acción $A_t$ en base a la distribución de densidad $P(\cdot)$
    \STATE Obtener recompensa $R_t$
    \STATE Actualizar la base empírica global $\bar{R} \leftarrow \bar{R} + \frac{1}{t}(R_t - \bar{R})$
    \FORALL{$a \in \{1, \dots, K\}$}
        \IF{$a = A_t$}
            \STATE $H(a) \leftarrow H(a) + \alpha(R_t - \bar{R})(1 - P(a))$
        \ELSE
            \STATE $H(a) \leftarrow H(a) - \alpha(R_t - \bar{R})P(a)$
        \ENDIF
    \ENDFOR
\ENDFOR
\end{algorithmic}
\end{algorithm}
```
