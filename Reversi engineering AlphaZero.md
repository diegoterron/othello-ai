---
title: Reversi engineering AlphaZero
author:
  - Aarón Mayoral Ansias
  - Diego Terrón Hernández
date: 2025-06-08
---

\begin{abstract}
 This project objective was to create an AlphaZero/AlphaGo type of artificial intelligence capable of playing the game Othello (also known as Reversi). Due to the limited compute power and academic focus, the methodology used is much less sophisticated than the Google DeepMind projects. 
 Nevertheless, a beginner level of playing was achieved, which is a good result taking into account the limited resources both in compute power and time.   
\end{abstract}

\begin{IEEEkeywords}
  Artificial intelligence, Neural network, Adversarial search, Reversi, Othello
\end{IEEEkeywords}


# Introduction

Othello is a simple board game for two players, played in an 8x8 board. Each player has initially  32 or 64 colored rocks. One player uses white rocks and the other one uses black ones. The objective of the game is having more rocks of your own color in the board than the adversary.  

The game starts with the configuration of rocks on the board show in \figurename  \ref{fig:initial-board}. The black-rocks player starts the game, placing a rock in a position where some white rock will be captured. In order for an opposite rock to be captured, it must be left between two enemy rocks in a straight line. This is, in \figurename\ref{fig:initial-board} the black player could only place a rock in C4, D3, E6 or F5. Diagonal captures are also allowed, but not possible from the initial state. \cite{wikipedia}

The game ends when all the board is covered with rocks or neither of the players has available moves. For this project interface, we used a tweaked set of rules that ends the game when a single player runs out of moves, but the AI was trained in the standard set of rules to avoid it learning to stall the opponent moves instead of actually playing the game.

\begin{figure}
  \centering
  \includegraphics{Initial_board}
  \caption{Initial State of the board in othello}
  \label{fig:initial-board}
\end{figure}

The AI consists in a Monte Carlo Tree Search (MCTS) algorithm, specifically an Upper Confidence Bound Tree (UCT). As will be later explained in \figurename\ref{fig:uct}, this algorithm uses a DefaultPolicy function, that we replaced for a neural network trained to predict the outcome of the match given a certain state of the board.

The structure for this document is as follows:

- A *Preliminar* section where techniques and related works will be further explored.
- A *Methodology* section where algorithms and tools used  will be explained.
- A *Results* section where the results of the project will be presented.
- A *Conclusion* section where the results will be analyzed.
- A *Bibliography*.


# Preliminar

En esta sección se hace una breve introducción de las técnicas empleadas y
también trabajos relacionados, si los hay.

## Employed methods

Describir aquí los métodos y técnicas empleadas (búsqueda en espacio de
estados, algoritmos genéticos, redes bayesianas, técnicas de clasificación,
redes neuronales, etc.). Si es necesario, separarlos en distintas subsecciones
dentro de Antecedentes.

Se pueden usar listas por puntos como sigue:

- Un punto: esto es un ejemplo de una lista.
- Otro punto.


Por último, se debe hacer un uso correcto de las referencias bibliográficas,
para que el lector pueda acceder a más información \cite{b2}. Todas las
referencias al final del documento deben ser citadas al menos una vez.


## Related Work

Se puede realizar un recorrido en la literatura sobre trabajos anteriores que
estén relacionados y que sea por tanto interesante comentar aquí. Por supuesto,
añadir las referencias bibliográficas correspondientes.


# Methodology

Esta sección se dedica a la descripción del método implementado en el trabajo.
Esta parte es la correspondiente a lo realmente desarrollado en el trabajo, y
se puede emplear pseudocódigo (nunca código), esquemas, tablas, etc.

\begin{figure}
  \centering
  \includegraphics{ejemplo}
  \caption{Ejemplo de un pie de figura. Imagen con derechos Creative Commons}
  \label{fig:ejemplo2}
\end{figure}

A continuación, un ejemplo de uso de listas numeradas:

1. *Trabajo con dos alumnos* poner nombre y
  apellidos completos de cada uno, y correos electrónicos de contacto (a ser
  posible de la Universidad de Sevilla). El orden de los alumnos se fijará por
  orden alfabético según los apellidos.
2. *Segundo título* cambiar la cabecera de la siguiente manera
	 1. solo se debe especificar un alumno.
	 2. la misma que la especificada en el
	    punto


Las figuras se deben mencionar en el texto, como la
\figurename \ref{fig:ejemplo}. También se pueden añadir ecuaciones, como la
ecuación \eqref{eq:ejemplo}.

\begin{equation}
  \label{eq:ejemplo}
  a + b = \gamma
\end{equation}

Un ejemplo de pseudocódigo se puede observar en la
\figurename \ref{pcd:mergesort}.

\begin{figure}
  \begin{pseudo}*
    \hd{\fn{mergesort}}(V) \\*
    \multicolumn{2}{l}{\textbf{Entrada}: un vector \( V \)} \\*
    \multicolumn{2}{l}{\textbf{Salida}: un vector con los elementos de \( V \)
      en orden} \\
    si \( V \) \textnormal{es unitario} entonces \\+
    devolver \( V \) \\-
    si no entonces \\+
    \( V_{1} \leftarrow \textnormal{primera mitad de } V\) \\
    \( V_{2} \leftarrow \textnormal{segunda mitad de } V\) \\
    \( V_{1} \leftarrow \pr{mergesort}(V_{1}) \) \\
    \( V_{2} \leftarrow \pr{mergesort}(V_{2}) \) \\-
    devolver \fn{mezcla}(V_{1}, V_{2})
  \end{pseudo}
  
  \begin{pseudo}*
    \hd{\fn{mezcla}}(V_{1}, V_{2}) \\*
    \multicolumn{2}{l}{%
      \textbf{Entrada}: dos vectores \( V_{1} \) y \( V_{2} \) ordenados
    } \\*
    \multicolumn{2}{l}{%
      \textbf{Salida}: un vector con los elementos de \( V_{1} \) y \( V_{2} \)
      en orden
    } \\
    si \( V_{1} \) \textnormal{no tiene elementos} entonces \\+
    devolver \( V_{2} \) \\-
    si no si \( V_{2} \) \textnormal{no tiene elementos} entonces \\+
    devolver \( V_{1} \) \\-
    si no entonces \\+
    \( x_{1} \leftarrow \textnormal{primer elemento de } V_{1} \) \\
    \( x_{2} \leftarrow \textnormal{primer elemento de } V_{2} \) \\
    si \( x_{1} \leq x_{2} \) entonces \\+
    \( x \leftarrow x_{1} \) \\
    \textnormal{quitar el primer elemento de} \( V_{1} \) \\-
    si no entonces \\+
    \( x \leftarrow x_{2} \) \\
    \textnormal{quitar el primer elemento de} \( V_{2} \) \\-
    \( V \leftarrow \fn{mezcla}(V_{1}, V_{2}) \) \\
    \textnormal{añadir} \( x \)
    \textnormal{como primer elemento de} \( V \) \\
    devolver \( V \)
  \end{pseudo}
  \caption{Algoritmo de ordenación \texttt{MergeSort}}
  \label{pcd:mergesort}
\end{figure}

# Results

En esta sección se detallarán tanto los experimentos realizados como los
resultados conseguidos:

-  Los experimentos realizados, indicando razonadamente la configuración
  empleada, qué se quiere determinar, y como se ha medido.
-  Los resultados obtenidos en cada experimento, explicando en cada caso lo
  que se ha conseguido.
-  Análisis de los resultados, haciendo comparativas y obteniendo
  conclusiones.


Se pueden hacer uso de tablas, como el ejemplo de la tabla \ref{tab:ejemplo}.

\begin{table}
  \caption{Ejemplo de tabla}
  \label{tab:ejemplo}
  \centering
  \begin{tabular}{ccc}
    \toprule
    A & B & C \\
    \midrule
    1 & 2 & 3 \\
    4 & 5 & 6 \\
    \bottomrule
  \end{tabular}
\end{table}


# Conclusion

Finalmente, se dedica la última sección para indicar las conclusiones obtenidas
del trabajo. Se puede dedicar un párrafo para realizar un resumen sucinto del
trabajo, con los experimentos y resultados. Seguidamente, uno o dos párrafos
con conclusiones. Se suele dedicar un párrafo final con ideas de mejora y
trabajo futuro.


\begin{thebibliography}{00}
\bibitem{wikipedia} https://es.wikipedia.org/wiki/Reversi
\bibitem{b1} G. Eason, B. Noble, and I. N. Sneddon, ``On certain integrals of Lipschitz-Hankel type involving products of Bessel functions,'' Phil. Trans. Roy. Soc. London, vol. A247, pp. 529--551, April 1955.
\bibitem{b2} J. Clerk Maxwell, A Treatise on Electricity and Magnetism, 3rd ed., vol. 2. Oxford: Clarendon, 1892, pp.68--73.
\bibitem{b3} I. S. Jacobs and C. P. Bean, ``Fine particles, thin films and exchange anisotropy,'' in Magnetism, vol. III, G. T. Rado and H. Suhl, Eds. New York: Academic, 1963, pp. 271--350.
\bibitem{b4} K. Elissa, ``Title of paper if known,'' unpublished.
\bibitem{b5} R. Nicole, ``Title of paper with only first word capitalized,'' J. Name Stand. Abbrev., in press.
\bibitem{b6} Y. Yorozu, M. Hirano, K. Oka, and Y. Tagawa, ``Electron spectroscopy studies on magneto-optical media and plastic substrate interface,'' IEEE Transl. J. Magn. Japan, vol. 2, pp. 740--741, August 1987 [Digests 9th Annual Conf. Magnetics Japan, p. 301, 1982].
\bibitem{b7} M. Young, The Technical Writer's Handbook. Mill Valley, CA: University Science, 1989.
\end{thebibliography}
