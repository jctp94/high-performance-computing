# Script — Video de Presentación
## Análisis de Calidad del Agua con Machine Learning
**Duración objetivo: 13–15 minutos | 19 diapositivas**

---

> **Instrucciones de uso:**
> - Cada sección está alineada a la diapositiva exacta del archivo `CADProyecto.pptx`.
> - El texto entre corchetes `[acción]` son indicaciones visuales, no se dicen en voz alta.
> - Ritmo recomendado: ~130 palabras por minuto, hablar despacio en secciones técnicas.
> - Las métricas en esta versión son los valores reales del notebook ejecutado.

---

## DIAPOSITIVA 1 — PORTADA
**⏱ 0:00 – 0:40**

Buenos días. Mi nombre es Juan Camilo Torres Peña y en este video presento el Proyecto II del curso de Computación de Alto Rendimiento.

El proyecto aborda el análisis de calidad del agua en ríos de la India usando Machine Learning. El stack tecnológico tiene dos capas bien diferenciadas: en la capa de procesamiento distribuido usamos Apache Spark con Hadoop HDFS; y en la capa de modelado, Keras con TensorFlow para la red neuronal y Scikit-Learn para los dos modelos de ensamblaje. En total, tres modelos complementarios que van desde la regresión continua hasta la clasificación operacional directa.

---

## DIAPOSITIVA 2 — PROBLEMA Y DATASET
**⏱ 0:40 – 2:00**

Esta diapositiva establece el problema y la materia prima con la que trabajamos.

Queremos predecir la calidad del agua de dos formas distintas. Primero, como un valor numérico continuo: el Índice de Calidad del Agua, o WQI, que va de 0 a 100 y más. Y segundo, como una categoría directa de acción: Excelente, Buena, Baja, Muy Baja o Inadecuada. Estas dos formas de predecir son complementarias y cada una tiene un caso de uso diferente, algo que veremos en detalle más adelante.

El dataset proviene del CPCB de India — el organismo nacional de control de contaminación — con 534 estaciones de monitoreo activas distribuidas en ríos de todo el país. El flujo de datos es el siguiente: las mediciones se almacenan en HDFS, el sistema de archivos distribuido de Hadoop, y desde ahí Apache Spark los carga y procesa de forma paralela.

La tabla de variables muestra los seis parámetros fisicoquímicos. Tres de ellos están marcados como de relevancia crítica o alta: el Oxígeno Disuelto, la Demanda Biológica de Oxígeno y los Coliformes Fecales. Estos tres parámetros van a aparecer repetidamente a lo largo de la presentación — los modelos van a confirmar de forma empírica que son los discriminadores más fuertes de la calidad del agua.

---

## DIAPOSITIVA 3 — PIPELINE ETL CON APACHE SPARK + HDFS
**⏱ 2:00 – 3:10**

Esta diapositiva muestra la arquitectura de procesamiento distribuido — el núcleo del proyecto desde la perspectiva del curso de Computación de Alto Rendimiento.

El flujo completo tiene seis pasos. HDFS almacena las mediciones con replicación de factor tres, lo que garantiza tolerancia a fallos. SparkSession carga los datos en el clúster de forma distribuida. A continuación se aplica la limpieza: filtro de nulos y conversión de tipos en DataFrames de Spark. Luego se calculan los sub-índices con UDFs y vistas SQL distribuidas. El WQI se obtiene como la suma ponderada de esos sub-índices. Y finalmente, la categoría de calidad se asigna por rangos usando Spark SQL.

Solo al final, cuando el dataset ya está limpio y transformado, se hace el `.toPandas()` al nodo driver para el entrenamiento local con sklearn y Keras.

Las cuatro decisiones de diseño de la parte inferior son las más importantes del proyecto desde el punto de vista de HPC. Primero: el ETL en Spark y el ML en sklearn no es una limitación — es la decisión correcta, porque Spark no está optimizado para la iteración de gradiente que requiere el entrenamiento de ML. Segundo: HDFS como fuente de datos desacopla el almacenamiento del cómputo. Tercero: `n_jobs=-1` en Random Forest aprovecha todos los núcleos disponibles para entrenar 300 árboles completamente en paralelo. Y cuarto: el `.toPandas()` solo al final minimiza la transferencia de datos entre workers y driver — principio básico de localidad de datos en sistemas distribuidos.

La clave del mensaje es esta: si mañana el dataset pasara de 534 estaciones a diez millones de registros diarios de sensores IoT, el código del ETL no cambia — solo se agregan nodos al clúster.

---

## DIAPOSITIVA 4 — ÍNDICE DE CALIDAD DEL AGUA (WQI)
**⏱ 3:10 – 4:00**

El WQI es el puente entre los parámetros crudos y las categorías de acción. Su cálculo tiene dos pasos.

Primero, cada parámetro se transforma a un sub-índice de 0 a 100 usando rangos de la literatura científica. Un oxígeno disuelto mayor o igual a 6 miligramos por litro obtiene 100 puntos — condición óptima. Coliformes fecales menores a 5 UFC por 100 mililitros también obtienen 100. Segundo, esos sub-índices se ponderan. Los pesos vienen de bibliografía especializada: Coliformes Fecales con 28.4%, Oxígeno Disuelto con 28.1% y Conductividad con 23.4% — los tres suman el 80% del índice total.

La escala de la derecha define las cinco categorías. Un WQI bajo, entre 0 y 25, significa agua excelente, apta sin tratamiento. Por encima de 100, el agua es inadecuada y activa una alerta crítica. Estas cinco categorías son exactamente las clases que los modelos de clasificación deben predecir directamente desde las mediciones crudas.

---

## DIAPOSITIVA 5 — EDA: DISTRIBUCIÓN ESPACIAL DEL WQI
**⏱ 4:00 – 4:45**

El mapa geográfico es la visualización más poderosa del proyecto. Con GeoPandas integrado al pipeline de Spark, pintamos el WQI promedio por estado usando una escala de color de verde a rojo.

El patrón tiene una lógica geográfica perfecta. En el norte, cerca del Himalaya, los ríos de montaña muestran los mejores índices — mínima intervención humana. En la zona central, la agricultura intensiva eleva los nitratos. Y en la cuenca del Ganges, especialmente en el corredor Varanasi-Kanpur, encontramos los peores índices del dataset — reflejo directo de la densidad urbana sin infraestructura de saneamiento adecuada.

---

## DIAPOSITIVA 6 — EDA: PARÁMETROS FISICOQUÍMICOS
**⏱ 4:45 – 5:20**

Esta diapositiva muestra tres visualizaciones del análisis exploratorio: la distribución de cada parámetro por categoría de calidad, la matriz de correlación entre variables, y las series temporales de DO y pH.

La distribución por categoría ya anticipa lo que los modelos van a aprender. Las cajas de DO y BOD se separan claramente entre las categorías Excelente e Inadecuada — son los parámetros más discriminativos visualmente. La matriz de correlación confirma las relaciones bioquímicas conocidas: alta correlación positiva entre DBO y Coliformes, y correlación negativa entre DO y DBO.

---

## DIAPOSITIVA 7 — EDA: HALLAZGOS CLAVE
**⏱ 5:20 – 6:10**

Esta tabla concentra las cinco observaciones del EDA que tienen implicaciones directas para el diseño de los modelos.

DO y BOD son los discriminadores más fuertes entre clases — y esto se va a confirmar después en la importancia de features de Random Forest y Gradient Boosting.

FECAL_COLIFORM tiene sesgo extremo: aproximadamente el 15% de las muestras tienen valores que en una escala normal serían outliers severos. Aquí es donde los modelos basados en árboles tienen una ventaja clara sobre la red neuronal — los árboles hacen comparaciones de orden, no de distancia, así que un valor de coliformes 1000 veces superior al promedio no distorsiona el modelo.

La correlación DBO–Coliformes mayor a 0.5 es real y tiene significado bioquímico. No se elimina por ser "redundante" — es información genuina sobre contaminación orgánica.

El hallazgo más importante para el diseño del entrenamiento es el último: la clase "Baja" domina el dataset con el 58% de las muestras. Sin corrección, cualquier modelo tendería a predecir siempre "Baja" para maximizar el accuracy. Por eso aplicamos `class_weight='balanced'` y validación cruzada estratificada.

---

## DIAPOSITIVA 8 — MODELO BASE: RED NEURONAL (ANN / KERAS)
**⏱ 6:10 – 7:20**

El primer modelo es la Red Neuronal implementada con Keras. Su tarea es regresión continua: predice el valor numérico del WQI.

La arquitectura tiene tres capas ocultas de 350 neuronas cada una con activación ReLU, y una capa de salida lineal con una sola neurona. El entrenamiento usa el optimizador Adam con learning rate de 0.001, función de pérdida de error cuadrático medio, 200 épocas y lotes de 16 muestras.

Los cuatro números destacados son los resultados sobre el conjunto de prueba — las muestras que el modelo nunca vio durante el entrenamiento. R² de 0.9989, RMSE de 0.51 puntos de WQI, MAE de 0.11 y sesgo de apenas 0.04. Son resultados casi perfectos.

Ahora bien, hay una advertencia fundamental que aparece al pie de la diapositiva. Las features de entrada no son las mediciones crudas del sensor. Son los sub-índices qr*, calculados aplicando exactamente las mismas reglas de dominio que producen el WQI. Es decir, la red está aprendiendo una combinación de variables que ya contienen implícitamente la respuesta. El R² cercano a 1 es matemáticamente esperado — no es una sorpresa.

Más importante: **en campo esos sub-índices no existen**. Solo existen las mediciones crudas del sensor. Para un sistema operacional real, necesitamos un enfoque diferente, y eso es exactamente lo que propone la siguiente diapositiva.

---

## DIAPOSITIVA 9 — REENCUADRE: REGRESIÓN → CLASIFICACIÓN DIRECTA
**⏱ 7:20 – 8:05**

Esta es la bisagra del proyecto. La tabla compara el enfoque original con el nuevo.

La red neuronal produce un número continuo del WQI, requiere sub-índices calculados manualmente como entrada y depende de conocimiento experto previo. Es útil para análisis retrospectivo de tendencias.

Los nuevos modelos producen una categoría de acción directa: Excelente, Baja, Inadecuada. Usan las mediciones crudas directas del sensor como entrada, sin ningún cálculo previo. Son útiles para alertas inmediatas en campo — el operador lee una etiqueta, no un número.

Para esta clasificación directa elegimos dos algoritmos: Random Forest, un ensemble paralelo con alta interpretabilidad; y Gradient Boosting, un ensemble secuencial que representa el estado del arte en precisión para datos tabulares.

---

## DIAPOSITIVA 10 — MODELO 1: RANDOM FOREST — ALGORITMO
**⏱ 8:05 – 9:10**

Random Forest introduce aleatoriedad en dos niveles para construir un conjunto de árboles diversos entre sí.

El primer nivel es el Bootstrap: cada árbol se entrena sobre aproximadamente el 63% de los datos, elegidos con reemplazo. El 37% restante, llamado Out-of-Bag, puede usarse como validación gratuita sin tocar el test set.

El segundo nivel es el Feature Bagging: en cada nodo de cada árbol, en lugar de evaluar las 6 features disponibles, se evalúan solo la raíz cuadrada de 6 — aproximadamente 2 — elegidas al azar. Esto garantiza que los 300 árboles sean estructuralmente distintos entre sí.

Al final, todos los árboles votan y gana la clase mayoritaria.

Los cuatro hiperparámetros clave de la tabla son: 300 estimadores para estabilizar el error; profundidad máxima de 12 para permitir reglas complejas sin memorización — el bagging controla el overfitting, no la profundidad; `class_weight='balanced'` para penalizar más los errores en las clases minoritarias; y `n_jobs=-1` para la paralelización total, que es exactamente el tipo de workload que aprovecha un entorno de alto desempeño — 300 árboles completamente independientes ejecutándose al mismo tiempo.

[Señalar el gráfico de importancias] La barra de Coliformes Fecales lidera con un 34.7% de importancia Gini, seguida de Oxígeno Disuelto con 27.3% y Conductividad con 13.7%. Esto coincide con los pesos del WQI original — el modelo aprendió empíricamente lo que la literatura científica ya sabía.

---

## DIAPOSITIVA 11 — RANDOM FOREST: RESULTADOS
**⏱ 9:10 – 10:00**

Los seis números del panel superior son los resultados de Random Forest sobre el conjunto de prueba.

Accuracy del 94.4% — el modelo acierta en más de 9 de cada 10 muestras. Precision macro del 71.3%, Recall macro del 72.6%, F1-Score macro del 71.9%. La diferencia entre accuracy y F1-macro es importante: la accuracy alta se debe en parte al dominio de la clase "Baja". El F1-macro, que trata todas las clases por igual, refleja mejor el desempeño real en el dataset desbalanceado.

AUC-ROC macro del 97.8% — prácticamente perfecta. Y el CV F1 medio de validación cruzada de 5 folds es 85.1% — esta es la métrica más confiable porque promedia el desempeño sobre cinco particiones distintas del dataset.

La nota al pie destaca un detalle interesante: la clase Excelente tiene el menor recall porque solo hay 14 muestras de esa clase en el dataset — con tan poco material de entrenamiento, el modelo tiene menos oportunidad de aprenderla. En contraste, el AUC-ROC de la clase Muy_Baja es perfecto: 1.000 — separación binaria perfecta.

---

## DIAPOSITIVA 12 — MODELO 2: GRADIENT BOOSTING — ALGORITMO
**⏱ 10:00 – 11:10**

Gradient Boosting tiene una filosofía radicalmente diferente. El diagrama muestra el proceso secuencial de cinco pasos.

Empieza con una predicción base — la clase más frecuente. Luego calcula los residuos: la diferencia entre la etiqueta real y la probabilidad predicha. Entrena un árbol pequeño y superficial para aprender esos residuos — el árbol no aprende el problema original, aprende qué corregir. Actualiza el ensemble sumando ese árbol multiplicado por el `learning_rate` de 0.05. Y repite hasta que la validación interna no mejora.

La analogía de los dardos lo resume bien: cada tiro corrige el error específico del anterior.

[Señalar la curva de convergencia] La curva muestra que el early stopping se activó en la ronda 152 — el modelo convergió antes de las 300 rondas configuradas. Eso es evidencia de que los hiperparámetros son correctos: el modelo encontró su punto óptimo y se detuvo solo.

Los cinco hiperparámetros clave: `learning_rate=0.05` — un valor bajo que hace que cada corrección sea pequeña y el ensemble sea robusto; `max_depth=4` para mantener los árboles como weak learners, porque el boosting amplifica los errores si los árboles son demasiado complejos; y `subsample=0.8` que activa el Gradient Boosting estocástico — cada árbol ve solo el 80% de las muestras, añadiendo diversidad y resistencia al overfitting.

La diferencia filosófica con Random Forest en una línea: RF reduce varianza con árboles paralelos profundos; GB reduce sesgo con árboles secuenciales superficiales.

---

## DIAPOSITIVA 13 — GRADIENT BOOSTING: RESULTADOS
**⏱ 11:10 – 11:55**

Los resultados de Gradient Boosting muestran un patrón interesante comparado con Random Forest.

Accuracy del 90.0% — cuatro puntos menos que RF. Precision macro 67.4%, Recall macro 68.7%, F1-macro 68.0%. A primera vista, RF parece mejor.

Pero las dos métricas más robustas cuentan una historia diferente. El CV F1 medio de validación cruzada es 87.9% para GB contra 85.1% para RF — Gradient Boosting generaliza mejor en promedio sobre distintas particiones del dataset. Y el AUC-ROC macro es 98.1% frente al 97.8% de RF — también mayor.

La explicación está en que el accuracy sobre un único test set puede ser sensible a cómo se distribuyeron las clases en ese split específico. El CV F1 de 5 folds es más confiable porque promedia sobre cinco splits distintos. Por eso GB es el modelo recomendado para producción.

La importancia de features de GB también confirma el mismo ranking: Coliformes Fecales encabeza con 39.0%, seguido de Oxígeno Disuelto — exactamente lo mismo que RF, pero con valores aún más concentrados en las dos variables dominantes.

---

## DIAPOSITIVA 14 — RANDOM FOREST VS. GRADIENT BOOSTING: ANÁLISIS COMPARATIVO
**⏱ 11:55 – 12:40**

Esta tabla pone cara a cara los dos modelos en los ocho criterios más relevantes.

En estrategia: RF es paralelo por bagging, GB es secuencial por boosting. En tipo de error: RF reduce varianza, GB reduce sesgo. En profundidad: RF usa árboles profundos porque el bagging controla el overfitting; GB usa árboles superficiales porque el boosting amplifica los errores complejos. En velocidad: RF gana claramente gracias a `n_jobs=-1`. En regularización: RF no tiene un mecanismo automático; GB tiene early stopping integrado.

Las dos últimas filas son las que deciden: CV F1 de 0.851 para RF versus 0.879 para GB. AUC-ROC de 0.978 versus 0.981.

El veredicto: RF para deployments donde la velocidad de entrenamiento y la interpretabilidad son críticas — por ejemplo, para explicar resultados a reguladores ambientales. GB para producción cuando la precisión de clasificación es lo más importante — decisiones de salud pública, alertas de contaminación.

---

## DIAPOSITIVA 15 — COMPARACIÓN GLOBAL: LOS TRES MODELOS
**⏱ 12:40 – 13:20**

La tabla de nueve criterios compara los tres modelos de forma unificada.

Los puntos más importantes son los siguientes. ANN requiere features derivadas que no existen en campo; RF y GB trabajan con mediciones crudas directas. ANN es sensible a outliers; los modelos de árboles son robustos por diseño. RF tiene la mayor interpretabilidad — las importancias de features son directamente comunicables a un regulador ambiental. GB tiene el mejor desempeño en tabular según la literatura y según nuestros propios resultados.

La última fila consolida las métricas finales: ANN con R²=0.9989, RF con Accuracy 94.4%, y GB con CV F1=0.879 y AUC=0.981.

El mensaje que está al pie de la tabla es el que cierra este bloque: los tres modelos son complementarios. La red neuronal rastrea la degradación continua del WQI a lo largo del tiempo. RF y GB detonan alertas inmediatas desde sensores en campo. No es una competencia — es una arquitectura de tres capas con roles diferenciados.

---

## DIAPOSITIVA 16 — ARQUITECTURA DE ALTO DESEMPEÑO: SPARK + HDFS
**⏱ 13:20 – 14:00**

Esta diapositiva explicita la arquitectura de dos capas que subyace a todo el proyecto.

La capa distribuida tiene HDFS y Spark trabajando juntos. HDFS aporta replicación de datos con factor tres, tolerancia a fallos automática y particionamiento por bloques. Spark aporta el DAG de transformaciones en memoria, el procesamiento distribuido y el SQL sobre RDDs. Esta capa maneja todo el ETL.

La capa local tiene sklearn ejecutando Random Forest con paralelización total — 300 árboles simultáneos en todos los núcleos del driver — y Keras con TensorFlow para la red neuronal, con capacidad de escalar a GPU si está disponible.

La transición entre capas es el `.toPandas()` — una sola operación que mueve los datos del clúster al driver ya limpios y transformados.

Esta separación es una decisión de arquitectura HPC: usar Spark para lo que hace mejor — ETL distribuido tolerante a fallos — y sklearn para lo que hace mejor — entrenamiento iterativo eficiente en memoria local.

---

## DIAPOSITIVA 17 — CONCLUSIONES
**⏱ 14:00 – 14:35**

Las conclusiones tienen dos bloques: uno sobre los modelos y otro sobre la arquitectura de alto desempeño.

Sobre los modelos: Gradient Boosting obtiene el mejor desempeño de clasificación con CV F1 de 0.879 y AUC de 0.981 — es el recomendado para producción. Random Forest es la mejor alternativa cuando se prioriza velocidad e interpretabilidad. La red neuronal complementa con la predicción continua del WQI. Y los tres modelos coinciden en identificar Coliformes Fecales, Oxígeno Disuelto y DBO como las variables más informativas — validación empírica perfecta de los pesos del índice de calidad.

Sobre la arquitectura HPC: Spark más HDFS demostró ser la elección correcta — el ETL escala sin modificaciones. La separación entre procesamiento distribuido y ML iterativo refleja el principio de usar la herramienta adecuada en cada capa. La paralelización nativa de Random Forest ilustra un algoritmo inherentemente diseñado para entornos de alto desempeño — 300 árboles independientes es un caso de paralelismo perfecto. Y la validación cruzada estratificada justifica su costo computacional adicional con una estimación de generalización mucho más confiable.

---

## DIAPOSITIVA 18 — TRABAJO FUTURO Y MEJORAS
**⏱ 14:35 – 15:00**

La tabla de trabajo futuro tiene siete líneas, pero quiero destacar las dos más importantes para este curso.

La primera es la optimización de hiperparámetros con Optuna y BayesSearchCV distribuido sobre Spark — con un costo de exploración menor que grid search, se espera una mejora de 2 a 5 puntos de F1.

La más relevante para el contexto de computación de alto rendimiento es la última: migrar el entrenamiento a **Spark MLlib**. Random Forest y Gradient Boosted Trees están implementados de forma nativa en MLlib con paralelización distribuida completa. Eso elimina el cuello de botella del `.toPandas()` al driver y permite que tanto el ETL como el entrenamiento ocurran en el mismo clúster — una arquitectura verdaderamente de alto desempeño de extremo a extremo.

---

## DIAPOSITIVA 19 — RESUMEN EJECUTIVO Y CIERRE
**⏱ 15:00 — fin**

[La diapositiva de cierre muestra el resumen ejecutivo completo en formato de tabla]

Para cerrar: construimos un pipeline completo que va desde datos estáticos en HDFS hasta tres modelos de Machine Learning con métricas de producción. La arquitectura de alto desempeño con Apache Spark y Hadoop HDFS garantiza escalabilidad sin modificar el código. Gradient Boosting es el modelo recomendado para aplicaciones operacionales de clasificación de calidad del agua. Y el siguiente paso natural es migrar a Spark MLlib para completar la arquitectura distribuida de extremo a extremo.

Muchas gracias por su atención.

---

## TABLA DE TIEMPOS

| Diapositiva | Contenido | Inicio | Fin | Duración |
|---|---|---|---|---|
| 1 | Portada | 0:00 | 0:40 | 40s |
| 2 | Problema y Dataset | 0:40 | 2:00 | 1m 20s |
| 3 | Pipeline ETL Spark + HDFS | 2:00 | 3:10 | 1m 10s |
| 4 | WQI — metodología | 3:10 | 4:00 | 50s |
| 5 | EDA — distribución espacial | 4:00 | 4:45 | 45s |
| 6 | EDA — parámetros fisicoquímicos | 4:45 | 5:20 | 35s |
| 7 | EDA — hallazgos clave | 5:20 | 6:10 | 50s |
| 8 | ANN — arquitectura y resultados | 6:10 | 7:20 | 1m 10s |
| 9 | Reencuadre regresión → clasificación | 7:20 | 8:05 | 45s |
| 10 | Random Forest — algoritmo | 8:05 | 9:10 | 1m 05s |
| 11 | Random Forest — resultados | 9:10 | 10:00 | 50s |
| 12 | Gradient Boosting — algoritmo | 10:00 | 11:10 | 1m 10s |
| 13 | Gradient Boosting — resultados | 11:10 | 11:55 | 45s |
| 14 | RF vs GB — comparativa | 11:55 | 12:40 | 45s |
| 15 | Comparación global 3 modelos | 12:40 | 13:20 | 40s |
| 16 | Arquitectura HPC Spark + HDFS | 13:20 | 14:00 | 40s |
| 17 | Conclusiones | 14:00 | 14:35 | 35s |
| 18 | Trabajo futuro | 14:35 | 15:00 | 25s |
| 19 | Resumen ejecutivo y cierre | 15:00 | 15:10 | 10s |
| **Total** | | | | **~15:10** |

---

> **Métricas reales incluidas en este script:**
> - ANN: R²=0.9989 · RMSE=0.51 · MAE=0.11 · Sesgo=0.04
> - Random Forest: Accuracy=94.4% · F1-macro=71.9% · AUC-ROC=97.8% · CV F1=85.1%
> - Gradient Boosting: Accuracy=90.0% · F1-macro=68.0% · AUC-ROC=98.1% · CV F1=87.9%
> - GB Early stopping: ronda 152 de 300
> - Feature importances RF: FC=34.7% · DO=27.3% · Conductividad=13.7%
> - Feature importances GB: FC=39.0% (lidera)

> **Consejos para la grabación:**
> - Practica las diapositivas 3, 12 y 17 por separado — son las más densas técnicamente.
> - En la diapositiva 13, haz una pausa antes de mencionar el CV F1 — es el momento donde GB supera a RF y merece énfasis.
> - En la diapositiva 17, separa claramente los dos bloques (modelos vs. arquitectura HPC) con un cambio de tono.
> - Al cambiar de diapositiva, deja 1 segundo de silencio para que el espectador lea el título.
> - La diapositiva 19 puede ir más lenta — es el cierre y merece énfasis final.
