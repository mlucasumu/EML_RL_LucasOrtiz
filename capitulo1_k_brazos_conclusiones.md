## 1.5. Conclusiones

### Resumen de Principales Resultados
Tras probar a fondo todos los algoritmos en el problema del bandido de k-brazos, sacamos una conclusión clara: no existe un algoritmo perfecto para todo. La mejor estrategia siempre dependerá del tipo de entorno y de cómo se comporten los premios.

- En entornos donde las recompensas varían de forma continua y lógica (como en la distribución *Normal*), métodos como **UCB1** y especialmente **UCB2** aprenden rapidísimo al elegir de forma inteligente qué opciones explorar. Sin embargo, **solo UCB1** logra mantener ese buen nivel en entornos algo más rígidos como el *Binomial*, ya que UCB2 se atasca por completo.
- Cuando nos enfrentamos a escenarios drásticos de "todo o nada" (recompensas *Bernoulli*), casi todos los algoritmos anteriores fallan de forma estrepitosa y acumulan pérdidas. En estos casos tan extremos, el ganador es el **Gradiente de Preferencias**. En vez de obsesionarse con calcular exactamente cuántos puntos da cada opción, este modelo se limita a comparar qué tan buena es una palanca respecto de las demás; gracias a esto, logra dominar el juego y no equivocarse casi nunca ($Regret \approx 0$).
- Para situaciones variadas o cuando no tenemos mucha potencia de cálculo, recurrir a los clásicos métodos semialeatorios (como **$\epsilon$-Decay** o directamente un $\epsilon$ fijo de $0.1$) resulta ser una estrategia buena.


### Limitaciones del Estudio
La limitación principal de los resultados reportados en este documento es que el juego asume que las reglas no cambian, la propiedad de **estacionalidad** del modelo K-Brazos abstraído. En el mundo real, la característica estocástica y las medias de recompensa ($\mu$) de una máquina o variable evolucionan y mutan con el paso del tiempo. Si transladamos estos algoritmos a la vida real, estos algoritmos podrían fracasar estrepitosamente porque se obcecarían en descartar opciones de escenarios que de repente podrían volverse muy rentables.

Además, este marco experimentacional excluye variables de "contexto" (variables climáticas, visuales o previas a la decisión) de cada tirada o instante $t$, tratando a las palancas bajo el paradigma probabilístico absoluto e idéntico en cada *step*.

### Impacto y Reflexión en el Aprendizaje por Refuerzo
Estudiar a fondo estas tres familias de algoritmos del bandido (apostar a lo seguro, investigar las opciones menos conocidas, o guiarse por las preferencias) no es para nada un ejercicio anticuado o puramente teórico. Al contrario: el concepto del *Arrepentimiento* y cómo minimizarlo son los cimientos más básicos que le dicen a cualquier Inteligencia Artificial cómo debe tomar sus decisiones.

Estas fórmulas e ideas matemáticas conforman el motor interno inicial de decisiones de cualquier agente moderno y complejo en el mundo real (como los sistemas de *Deep Reinforcement Learning* de las IA generativas o de los coches autónomos).

### Líneas Futuras de Estudio
El siguiente paso natural para avanzar en este campo sería crear simulaciones de **Bandidos No Estacionarios (*Non-stationary Bandits*)**, es decir, enseñar a la Inteligencia Artificial a jugar en entornos donde las reglas y los premios van cambiando con el tiempo. Para lograrlo, habría que dotar a los algoritmos con capacidades de adaptación "olvidadizas": mecanismos matemáticos que les permitan ignorar lecciones antiguas que ya no sirven y centrarse en lo que está ocurriendo ahora mismo.

Más adelante, el objetivo final sería evolucionar hacia **Bandidos Contextuales (*Contextual Bandits*)**. En este escenario, la IA no jugaría a ciegas, sino que recibiría "pistas" del entorno antes de tener que elegir una opción (por ejemplo, saber qué tiempo hace antes de recomendar un tipo de ropa). Esto abre directamente las puertas hacia sistemas de inteligencia artificial mucho más completos y complejos que los que hemos visto hasta ahora.
