# Capítulo 1: Problema del Bandido de *k*-brazos

## 1.1. Introducción

### Descripción del Problema y Relevancia
El problema del bandido de $k$-brazos (o *Multi-Armed Bandit*) ilustra formalmente el dilema fundamental del Aprendizaje por Refuerzo: el balance iterativo entre la exploración y la explotación. Un agente computacional se enfrenta a un entorno estacionario y estocástico modelado como una máquina tragaperras con $k$ palancas (brazos), cada una ocultando una distribución de probabilidad de recompensa subyacente diferente y desconocida *a priori*. 

Con un tiempo límite o presupuesto de $T$ interacciones, la meta del agente es maximizar su recompensa acumulativa. La toma de su decisión es crítica: si confía únicamente en la palanca que le ha concedido mayor ganancia económica hasta ahora (explotación), recae en el riesgo de converger hacia una recompensa local ignorando una palanca numéricamente superior. Si, por el contrario, invierte un esfuerzo desmedido en probar nuevas palancas aleatorias para descubrir el verdadero margen de beneficio (exploración), desperdicia iteraciones cobrando recompensas bajas con la consecuente penalización temporal. 

### Motivación del Trabajo
La razón principal por la que estudiamos este problema es porque se parece a muchas decisiones prácticas que tomamos en el mundo real (como elegir en qué anuncio invertir dinero o qué tratamiento médico probar). Es un entorno "simple" porque las decisiones que tomamos ahora no cambian las opciones que tendremos en el futuro (cada tirada es independiente). Sin embargo, estudiarlo es clave para entender cómo funciona la inteligencia artificial cuando aprende a base de prueba y error. 

Si logramos comprender cómo estos algoritmos básicos ($\epsilon$-Greedy o UCB) interactúan con la duda de *"¿me quedo con lo que ya sé que funciona o arriesgo a probar algo nuevo?"*, habremos construido los pilares necesarios para entender a los verdaderos agentes robóticos complejos que resolverán problemas mucho más difíciles en el futuro.

### Objetivos del Informe
El desarrollo de esta primera fase trata de cumplir los siguientes objetivos técnicos:
1. Diseñar y programar un entorno modular en Python capaz de modelar un Bandido bajo tres tipologías de distribución de recompensa continuas y discretas: Normal, Binomial y Bernoulli. 
2. Implementar desde cero tres familias clásicas de algoritmos de toma de decisiones: Explotación estocástica ($\epsilon$-Greedy y sus variantes con decaimiento), Ascenso de Gradiente (Softmax y métricas de Preferencia) y Optmismo basado en incertidumbre (UCB-1, UCB-2 y UCB1-Tuned).
3. Evaluar dichas familias sometiéndolas empíricamente frente al entorno mediante $N=500$ ejecuciones independientes simulando $T=300$ pasos temporales.
4. Analizar la escalabilidad, la ratio de acierto de selección óptima y la mitigación del arrepentimiento acumulado (*Cumulative Regret*).

### Organización del Documento
Para facilitar su asimilación, este primer capítulo estructura su narrativa en el siguiente formato secuencial. En la sección de **Desarrollo** (1.2), se sentarán las bases teóricas definiendo matricial y probabilísticamente las distribuciones a las que se enfrenta el bandido, adjuntando a la par un breve estado del arte de las influencias bibliográficas. Seguidamente, en la sección de **Algoritmos** (1.3), se expondrá la matemática formal y el pseudocódigo subyacente de cada política heurística testeada. Las asunciones y las simulaciones se materializan descriptivamente de la mano de gráficas de rendimiento en la sección referente a la **Evaluación/Experimentos** (1.4). Por último, las métricas son discutidas críticamente en las **Conclusiones** (1.5) para certificar el descubrimiento de los parámetros ganadores globales.
