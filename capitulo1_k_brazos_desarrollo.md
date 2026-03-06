## 1.2. Desarrollo

### Contexto y Antecedentes
El dilema de exploración frente a explotación es una encrucijadas elemental dentro del Aprendizaje por Refuerzo. Mientras que en el aprendizaje supervisado un algoritmo recibe explícitamente la respuesta correcta a modo de etiqueta, en el aprendizaje evaluativo (característico del RL) un agente debe descubrir autónomamente qué acciones brindan la mayor ganancia probándolas. El modelo del *Multi-Armed Bandit* (Bandido Multibrazo) es el paradigma para aislar y estudiar estas propiedades evaluativas y de decisión en un entorno estacionario y no asociativo (es decir, donde el entorno no cambia de estado). Comprender la balanza entre arriesgar ganancias seguras presentes y el potencial de ganancias futuras desconocidas cimienta la matemática que los agentes complejos aplican cuando operan con los valores $Q$ y variables de estado multidimensionales.

### Definición Formal del Problema
El entorno estocástico se define bajo el espacio de recompensas discretas o continuas para un conjunto finito de acciones o brazos $a \in \mathcal{A} = \{1, 2, \dots, k\}$. 

El verdadero y desconocido "valor esperado" de elegir una acción $a$ se denota habitualmente como el valor a maximizar $q_*(a)$, y está formulado como la esperanza matemática de la recompensa en un instante $t$:
$$ q_*(a) = \mathbb{E}[R_t \mid A_t = a] $$

A lo largo del presente marco de validación, las recompensas $R_t$ derivan su valor estocástico basándose en tres entornos implementados:

1. **Entorno de Recompensas Normales (Gausianas)**
   La retribución estocástica de cada acción se muestrea de una distribución diferencial $R_t \sim \mathcal{N}(\mu_a, \sigma_a^2)$, donde la verdadera medida subyacente de recompensa está dominada por su función de densidad de probabilidad (PDF):
   $$ f(x \mid \mu_a, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu_a)^2}{2\sigma^2}} $$

2. **Entorno de Recompensas Binomiales**
   En este escenario, la activación del brazo $a$ devuelve una recompensa entera discreta ($x \in \mathbb{N}$) que contabiliza la cantidad total de éxitos sobre $n_a$ ensayos independientes. Es dirigido por la variable de masa probabilística $R_t \sim B(n_a, p_a)$:
   $$ P(R_t=x) = \binom{n_a}{x} p_a^x (1 - p_a)^{n_a - x} $$

3. **Entorno de Recompensas de Bernoulli**
   Un escenario más sencillo donde el resultado de tirar de la palanca solo puede ser un éxito o un fracaso absoluto ($R_t \in \{0, 1\}$). Modela la variable $R_t \sim Bernoulli(p_a)$ en donde el valor promedio de iteración a estabilizar es exactamente la probabilidad intrínseca al brazo evaluado ($q_*(a) = p_a$):
   $$ P(R_t) = p_a^{R_t} (1 - p_a)^{1 - R_t} $$

### Enfoque y Justificación Metodológica
Para abordar el problema estocástico, se ha abstraído un *framework* orientado a objetos que desacopla completamente el "Entorno" del "Agente". Esto permite insertar módulos de políticas que basan su decisión algorítmica interactuando con las mismas estructuras pre-configuradas $R_t$. El presente trabajo se caracteriza por implementar la comparativa simultánea y simétrica del estado del arte (*$\epsilon$-Greedy* frente al Límite de Confianza *UCB* y el cálculo heurístico *Softmax*) bajo un volumen estadístico ($N=500$ iteraciones promediadas), garantizando un estudio numérico de optimización con mitigación del *Arrepentimiento Acumulado* en ecosistemas tanto lógicos continuos como discretos.

### 1.2.1. Trabajos Relacionados
El planteamiento algorítmico, su formulación paramétrica y las heurísticas estandarizadas analizadas en este proyecto se fundamentan y se derivan de la literatura existente en el campo del Aprendizaje por Refuerzo:
- **Métodos Epsilon y Ascenso de Gradiente (Sutton y Barto, 2018):** El diseño de las estrategias $\epsilon$-Greedy y de los métodos probabilísticos como el *Ascenso de Gradiente* se ha construido siguiendo las reglas y fundamentos explicados en el libro de referencia "Reinforcement Learning: An Introduction" \cite{sutton2018reinforcement}.
- **Manejo de la Incertidumbre con UCB (Auer et al., 2002):** Para programar los algoritmos UCB (Upper Confidence Bound) de forma robusta y entender cómo utilizan sus fórmulas matemáticas para explorar las opciones menos conocidas, nos hemos apoyado en el estudio "Finite-time Analysis of the Multiarmed Bandit Problem" \cite{auer2002finitetime}.
