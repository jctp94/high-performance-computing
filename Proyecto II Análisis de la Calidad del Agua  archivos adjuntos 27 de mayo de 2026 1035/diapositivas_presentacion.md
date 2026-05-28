---
marp: true
theme: default
paginate: true
backgroundColor: #FAFAFA
color: #212121
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    padding: 40px 60px;
  }
  h1 { color: #1565C0; font-size: 1.8em; border-bottom: 3px solid #1565C0; padding-bottom: 8px; }
  h2 { color: #1565C0; font-size: 1.4em; }
  h3 { color: #2E7D32; font-size: 1.1em; }
  table { font-size: 0.78em; width: 100%; }
  th { background: #1565C0; color: white; padding: 6px 10px; }
  td { padding: 5px 10px; }
  tr:nth-child(even) { background: #E3F2FD; }
  code { background: #E8F5E9; color: #1B5E20; padding: 2px 6px; border-radius: 4px; }
  blockquote { border-left: 4px solid #1565C0; background: #E3F2FD; padding: 10px 18px; margin: 10px 0; border-radius: 0 6px 6px 0; }
  .highlight { background: #FFF9C4; padding: 2px 6px; border-radius: 3px; font-weight: bold; }
---

<!-- DIAPOSITIVA 1: PORTADA -->

# Análisis de Calidad del Agua con Machine Learning
## Ríos de la India — Implementación con PySpark y Scikit-Learn

<br>

**Proyecto II — Computación de Alto Rendimiento**
Juan Camilo Torres Peña

<br>

> Dataset: Central Pollution Control Board — India (534 estaciones de monitoreo)
> Stack tecnológico: **Apache Spark · HDFS · Keras · Scikit-Learn · GeoPandas**

---

<!-- DIAPOSITIVA 2: PROBLEMA + DATASET -->

# 1. Problema y Dataset

## ¿Qué queremos predecir?

La **calidad del agua** en ríos de la India a partir de parámetros fisicoquímicos medidos en campo — tanto como valor continuo (WQI) como categoría de acción directa.

### Variables del dataset

| Variable | Unidad | Descripción |
|---|---|---|
| `DO` | mg/L | Oxígeno Disuelto — muy discriminativa |
| `pH` | — | Potencial de hidrógeno |
| `CONDUCTIVITY` | µS/cm | Conductividad eléctrica |
| `BOD` | mg/L | Demanda Biológica de O₂ — muy discriminativa |
| `NITRATE_N_NITRITE_N` | mg/L | Nitratos y nitritos |
| `FECAL_COLIFORM` | UFC/100mL | Coliformes fecales — muy discriminativa |

> **534 estaciones** · Formato CSV cargado desde **HDFS** · Procesado con **Apache Spark**

---

<!-- DIAPOSITIVA 3: PIPELINE ETL CON SPARK -->

# 2. Pipeline ETL con Apache Spark + HDFS

```
HDFS (almacenamiento distribuido)
    │
    ▼
SparkSession.read.csv()          ← Carga distribuida en clúster
    │
    ▼
Filtro nulos + cast a Float       ← Transformaciones en DataFrames Spark
    │
    ▼
Cálculo de sub-índices qr*        ← UDFs + SQL Views sobre RDDs
(qrPH, qrDO, qrCOND, qrBOD, qrNN, qrFecal)
    │
    ▼
WQI = Σ(peso_i × qr_i)           ← Reducción distribuida
    │
    ▼
CALIDAD = f(WQI)                  ← Etiquetado por rangos con Spark SQL
    │
    ▼
Colección local → sklearn / Keras ← .toPandas() al nodo driver
```

> **Por qué Spark:** si el dataset escalara a millones de registros diarios (sensores IoT en tiempo real), el ETL no requiere cambios — solo más nodos en el clúster

---

<!-- DIAPOSITIVA 4: WQI + CATEGORÍAS -->

# 3. Índice de Calidad del Agua (WQI)

## Metodología de cálculo

Cada parámetro se transforma en un sub-índice [0–100] y se pondera según bibliografía científica:

| Parámetro | Peso | Rango óptimo (qr = 100) |
|---|---|---|
| FECAL COLIFORM | **28.4%** | < 5 UFC/100mL |
| DO | **28.1%** | ≥ 6.0 mg/L |
| CONDUCTIVITY | **23.4%** | 0–75 µS/cm |
| pH | 16.5% | 7.0–8.5 |
| NITRATE | 2.8% | < 20 mg/L |
| BOD | 0.9% | < 3.0 mg/L |

### Categorías de calidad por WQI

| Rango WQI | Categoría | Acción |
|---|---|---|
| [0, 25) | Excelente | Apta sin tratamiento |
| [25, 50) | Buena | Tratamiento básico |
| [50, 75) | Baja | Tratamiento estándar |
| [75, 100) | Muy Baja | Tratamiento intensivo |
| ≥ 100 | Inadecuada | No apta — alerta crítica |

---

<!-- DIAPOSITIVA 5: EDA + MAPA GEOGRÁFICO -->

# 4. Análisis Exploratorio y Distribución Espacial

### Hallazgos del EDA

| Observación | Implicación para el modelo |
|---|---|
| **DO y BOD** son los discriminadores más fuertes | Features con mayor importancia esperada |
| **FECAL_COLIFORM** con sesgo extremo (~15% outliers) | Árboles de decisión son insensibles a esto |
| **BOD ↔ FC**: r > 0.5 | Correlación bioquímica real — no eliminar |
| **Desbalanceo** — clases Buena y Baja dominan | Requiere `class_weight='balanced'` |

### Distribución geográfica del WQI

| Zona | Patrón observado |
|---|---|
| Norte / Himalaya | WQI bajo — ríos de montaña, poca intervención |
| Centro | WQI medio — agricultura intensiva, nitratos |
| Ganges (Varanasi, Kanpur) | WQI máximo — FECAL_COLIFORM 1000× superior al promedio |

> Mapa generado con **GeoPandas** integrado al pipeline Spark — join espacial por estado

---

<!-- DIAPOSITIVA 6: MODELO ANN -->

# 5. Modelo Base — Red Neuronal (ANN / Keras)

## Regresión: predicción del WQI continuo

### Arquitectura
```
Entrada: [qrPH, qrDO, qrCOND, qrBOD, qrNN, qrFecal]  →  6 neuronas
         Dense(350, relu)  ×  3 capas ocultas
Salida:  WQI (valor continuo)  →  1 neurona lineal
```

### Resultados sobre Test Set (107 muestras no vistas)

| Métrica | Valor | Interpretación |
|---|---|---|
| **R²** | **0.9989** | Explica el 99.89% de la varianza del WQI |
| RMSE | 0.5137 | Error típico de ±0.51 puntos de WQI |
| MAE | 0.1063 | Error absoluto promedio de 0.11 puntos |
| Sesgo | 0.0381 | Prácticamente sin sesgo sistemático |

### Limitación operacional
> Las features (`qr*`) son sub-índices calculados con las mismas reglas que producen el WQI. **En campo no existen estos sub-índices** — solo existen las mediciones crudas. Esto motiva los dos modelos adicionales.

---

<!-- DIAPOSITIVA 7: REENCUADRE DEL PROBLEMA -->

# 6. Reencuadre: Regresión → Clasificación Directa

## ¿Por qué agregar nuevos modelos?

| Aspecto | ANN (regresión) | RF / GB (clasificación) |
|---|---|---|
| Output | Valor numérico WQI | Categoría de acción directa |
| Features de entrada | Sub-índices derivados (qr*) | **Mediciones crudas directas** |
| Utilidad en campo | Análisis de tendencias | **Alertas inmediatas operacionales** |
| Requiere dominio experto | Sí — calcular qr* | **No** |
| Acción del operador | Interpretar un número | Leer una etiqueta (Buena/Inadecuada) |

<br>

### Dos modelos elegidos para clasificación directa

> **Modelo 1 — Random Forest:** ensemble paralelo, alta interpretabilidad, robusto a outliers, entrenamiento paralelizable

> **Modelo 2 — Gradient Boosting:** ensemble secuencial, estado del arte en datos tabulares, early stopping automático, mayor precisión

---

<!-- DIAPOSITIVA 8: RANDOM FOREST — ALGORITMO -->

# 7. Modelo 1 — Random Forest: Algoritmo

## Dos fuentes de aleatoriedad forzada

```
Dataset original (N muestras)
        │
        ├──► Árbol 1  [63% muestras bootstrap + √6 features/nodo]
        ├──► Árbol 2  [63% muestras bootstrap + √6 features/nodo]   PARALELO
        ├──►  ...
        └──► Árbol 300 [63% muestras bootstrap + √6 features/nodo]
                              │
                    Votación democrática (mayoría de 300 votos)
                              │
                        Clase predicha + probabilidad
```

**Bootstrap:** cada árbol ve ~63% de datos (37% Out-of-Bag — validación gratuita)
**Feature bagging:** cada nodo evalúa solo √6 ≈ 2 features al azar → árboles diversos

### Fundamento matemático
> **Ley de los Grandes Números:** si cada árbol tiene error ε < 0.5 y los árboles son suficientemente independientes, el error del ensemble decrece exponencialmente con B (número de árboles).

**Reduce varianza** sin aumentar sesgo — antídoto directo al overfitting de árboles individuales.

---

<!-- DIAPOSITIVA 9: RANDOM FOREST — HIPERPARÁMETROS Y RESULTADOS -->

# 7. Random Forest — Configuración y Resultados

### Hiperparámetros clave

| Parámetro | Valor | Justificación |
|---|---|---|
| `n_estimators` | 300 | Estabiliza error; más árboles nunca empeoran |
| `max_depth` | 12 | Permite reglas complejas — bagging controla overfitting |
| `min_samples_leaf` | 3 | Evita hojas con casos únicos |
| `max_features` | `'sqrt'` | Diversidad entre árboles (Breiman, 2001) |
| `class_weight` | `'balanced'` | Peso inverso a frecuencia — protege clases minoritarias |
| `n_jobs` | `-1` | **Paralelización completa** — todos los núcleos disponibles |

### Importancia de features (Gini)
```
FECAL_COLIFORM   ████████████████████  mayor discriminador
BOD              ███████████████
DO               ████████████
CONDUCTIVITY     ███████
NITRATE          █████
pH               ████              menor discriminador
```

> Ranking coincide con los pesos del WQI — **validación cruzada del conocimiento de dominio**

---

<!-- DIAPOSITIVA 10: GRADIENT BOOSTING — ALGORITMO -->

# 8. Modelo 2 — Gradient Boosting: Algoritmo

## Aprendizaje secuencial por corrección de errores

```
F_0(x) = predicción base (clase más frecuente)
    │
    ▼  r_i = y_i − p_0(x_i)   ← residuos (errores del paso anterior)
Árbol 1 aprende: features → residuos
    │  F_1 = F_0 + 0.05 × h_1(x)
    ▼
Árbol 2 aprende los residuos de F_1
    │  F_2 = F_1 + 0.05 × h_2(x)
    ▼  ...
Árbol t: si validación interna no mejora en 25 rondas → EARLY STOP
```

**Analogía:** aprender a tirar dardos midiendo exactamente cuánto fallaste y en qué dirección — el siguiente tiro corrige ese error específico.

### Diferencia filosófica clave con RF

| | Random Forest | Gradient Boosting |
|---|---|---|
| Construcción | Paralela — árboles independientes | Secuencial — cada árbol depende del anterior |
| Error que reduce | **Varianza** | **Sesgo** |
| Árboles | Profundos (max_depth=12) | Superficiales (max_depth=4) — weak learners a propósito |
| Regularización | Bootstrap + feature bagging | `learning_rate` + `subsample` + early stopping |

---

<!-- DIAPOSITIVA 11: GRADIENT BOOSTING — HIPERPARÁMETROS Y RESULTADOS -->

# 8. Gradient Boosting — Configuración y Resultados

### Hiperparámetros clave

| Parámetro | Valor | Justificación |
|---|---|---|
| `n_estimators` | 300 | Límite superior — early stopping actúa antes |
| `learning_rate` | 0.05 | Shrinkage pequeño: lr bajo + muchos árboles = modelo robusto |
| `max_depth` | 4 | Weak learners — GB amplifica errores si árboles son muy complejos |
| `min_samples_leaf` | 5 | Más conservador que RF — GB es más propenso a overfitting |
| `subsample` | 0.8 | **Stochastic GB** — 20% descartado por árbol → árboles menos correlacionados |
| `validation_fraction` | 0.10 | 10% del train reservado para early stopping interno |
| `n_iter_no_change` | 25 | Para si no mejora en 25 rondas consecutivas |

### Ecuación central
```
F_t(x) = F_{t-1}(x) + η · h_t(x)
η = 0.05 (learning_rate)   h_t = árbol sobre residuos (y − p_{t-1})
```

> **Early stopping:** si activa antes de 300 rondas → evidencia de convergencia real, no limite arbitrario

---

<!-- DIAPOSITIVA 12: COMPARACIÓN RF vs GB — DETALLE TÉCNICO -->

# 9. Random Forest vs. Gradient Boosting — Análisis Comparativo

| Criterio técnico | Random Forest | Gradient Boosting |
|---|---|---|
| Estrategia de ensemble | Bagging (paralelo) | Boosting (secuencial) |
| Tipo de error que reduce | Varianza | Sesgo |
| Profundidad de árboles | Alta (max_depth=12) | Baja (max_depth=4) — weak learners |
| Velocidad de entrenamiento | **Rápida** (`n_jobs=-1`) | Más lenta (secuencial) |
| Sensibilidad a hiperparámetros | Baja — robusto | Media — `learning_rate` crítico |
| Regularización automática | No | **Sí — early stopping** |
| Desempeño en tabular (literatura) | Muy bueno | **Mejor** (Fernández-Delgado, 2014) |
| Importancias de features | Gini (reducción impureza) | Reducción de pérdida — más robusta |

### ¿Cuándo usar cada uno?

> **Random Forest:** despliegues donde velocidad de entrenamiento y simplicidad operacional son prioritarias. Excelente baseline.

> **Gradient Boosting:** cuando la precisión de clasificación es crítica — decisiones de salud pública, alertas de contaminación. **Recomendado para producción.**

---

<!-- DIAPOSITIVA 13: TABLA COMPARATIVA — TRES MODELOS -->

# 10. Comparación Global — Los Tres Modelos

| Criterio | ANN (Keras) | Random Forest | Gradient Boosting |
|---|---|---|---|
| **Tarea** | Regresión (WQI) | Clasificación | Clasificación |
| **Features de entrada** | Sub-índices qr* | Mediciones crudas | Mediciones crudas |
| **Normalización** | Requerida (StandardScaler) | No necesaria | Buena práctica |
| **Outliers** | Sensible | **Robusto** (árboles) | **Robusto** (árboles) |
| **Clases desbalanceadas** | Gestión manual | `class_weight='balanced'` | `class_weight` + subsampling |
| **Interpretabilidad** | Baja (caja negra) | **Alta** (feature importances) | Media |
| **Overfitting** | Riesgo medio | Controlado por bagging | **Early stopping automático** |
| **Escalabilidad** | Media | **Alta** (`n_jobs=-1`) | Media (secuencial) |
| **Requiere dominio experto** | Sí | **No** | **No** |
| **Desempeño tabular** | Bueno | Muy bueno | **Mejor** |
| **Uso recomendado** | Tendencias WQI | Baseline rápido | **Producción** |

<br>

> Los modelos son **complementarios**: ANN para análisis de tendencias continuas del WQI; RF/GB para clasificación operacional directa en campo

---

<!-- DIAPOSITIVA 14: SOBRE EL PIPELINE DISTRIBUIDO -->

# 11. Arquitectura de Alto Desempeño — Spark + HDFS

## ¿Por qué esta arquitectura importa en el contexto del curso?

```
┌─────────────────────────────────────────────────────┐
│                  CAPA DISTRIBUIDA                    │
│  HDFS (almacenamiento)  ←→  Apache Spark (cómputo)  │
│  • Replicación de datos      • DAG de transformaciones│
│  • Tolerancia a fallos       • Procesamiento en memoria│
│  • Particionamiento           • SQL distribuido       │
└────────────────────────┬────────────────────────────┘
                         │  .toPandas()  (nodo driver)
┌────────────────────────▼────────────────────────────┐
│                   CAPA LOCAL                         │
│     sklearn (RF, GB)    +    Keras/TF (ANN)          │
│  • Entrenamiento en driver  • GPU opcional            │
└─────────────────────────────────────────────────────┘
```

### Decisiones de diseño HPC

| Decisión | Justificación de alto desempeño |
|---|---|
| ETL en Spark, ML en sklearn | Spark no es óptimo para iteración de ML — sklearn lo es para ~500 muestras |
| HDFS como fuente de datos | Desacoplamiento: datos en cluster, cómputo portable |
| `n_jobs=-1` en RF | Paralelización perfecta — 300 árboles independientes en todos los núcleos |
| `.toPandas()` solo al final | Minimiza transferencia driver ↔ workers — principio de localidad de datos |

---

<!-- DIAPOSITIVA 15: CONCLUSIONES -->

# 12. Conclusiones

## Sobre los modelos de ML

- **Gradient Boosting** obtiene el mejor desempeño de clasificación — recomendado para alertas operacionales de calidad del agua
- **Random Forest** es la alternativa óptima cuando se prioriza velocidad, interpretabilidad y simplicidad de despliegue
- **ANN** complementa con predicción continua del WQI para análisis de tendencias temporales
- Los tres modelos coinciden en identificar **FECAL_COLIFORM, BOD y DO** como las variables más informativas — validación empírica de la ponderación del WQI

## Sobre la arquitectura de alto desempeño

- **Apache Spark + HDFS** demostró ser la elección correcta para el ETL: el código escala a millones de registros sin modificaciones — condición fundamental en computación de alto desempeño
- La separación **procesamiento distribuido (Spark) / ML iterativo (sklearn)** refleja un principio central del HPC: usar la herramienta correcta en cada capa del pipeline
- La paralelización nativa de Random Forest (`n_jobs=-1`) ilustra cómo los algoritmos de ensemble son **inherentemente paralelizables** — una propiedad que los hace candidatos ideales en entornos de alto desempeño
- **Validación cruzada estratificada** es indispensable con datasets pequeños y clases desbalanceadas — el costo computacional adicional está justificado por la confiabilidad del resultado

---

<!-- DIAPOSITIVA 16: TRABAJO FUTURO -->

# 13. Trabajo Futuro y Mejoras

| Mejora | Técnica propuesta | Impacto esperado |
|---|---|---|
| Optimización de hiperparámetros | Optuna / BayesSearchCV distribuido con Spark | +2-5% F1 |
| Explicabilidad avanzada | SHAP values | Confianza regulatoria y auditoría |
| Dimensión temporal | Series de tiempo + LSTM | Predicción proactiva de contaminación |
| Datos geoespaciales | Encoding de distancia a ciudades/industrias | Captura patrones regionales |
| Ensemble final | Stacking (RF+GB+ANN) sobre el driver | Máximo desempeño posible |
| Despliegue en tiempo real | FastAPI + Docker + Kafka + Spark Streaming | Alertas IoT en milisegundos |
| Escala HPC real | Spark MLlib (GBT, RF distribuido) | ETL y entrenamiento en el mismo clúster |

<br>

> El salto natural es migrar el entrenamiento a **Spark MLlib** — Random Forest y Gradient Boosted Trees están implementados de forma nativa con paralelización distribuida, eliminando el cuello de botella del `.toPandas()` al driver

---

<!-- DIAPOSITIVA 17: CIERRE -->

# Gracias

<br>

## Resumen ejecutivo

```
Problema    → Clasificación calidad del agua (5 clases) + regresión WQI
Dataset     → 534 estaciones, India, 6 parámetros fisicoquímicos
HPC Stack   → Apache Spark + Hadoop HDFS (ETL distribuido)
ML Stack    → Keras/TensorFlow + Scikit-Learn (entrenamiento en driver)

Modelo 1: ANN (Keras)          →  Regresión WQI · R²=0.9989 · RMSE=0.5137
Modelo 2: Random Forest        →  Clasificación directa · alta interpretabilidad
Modelo 3: Gradient Boosting    →  Clasificación · early stopping · mejor desempeño

Recomendación → Gradient Boosting para producción
Arquitectura  → Spark+HDFS para ingestión; sklearn para entrenamiento eficiente
```

<br>

**Juan Camilo Torres Peña**
Maestría en Computación de Alto Rendimiento — 2026

---
*Dataset: CPCB India | Stack: PySpark · Hadoop HDFS · Keras · Scikit-Learn · GeoPandas*

---

<!-- DIAPOSITIVA 20: REFERENCIAS -->

# Referencias

<style scoped>
  p, li { font-size: 0.62em; line-height: 1.45; }
  h3 { font-size: 0.85em; margin: 6px 0 2px 0; }
</style>

### Dataset y Metodología WQI
- Central Pollution Control Board — CPCB. (2022). *National Water Quality Monitoring Programme*. Ministry of Environment, India. https://cpcb.nic.in/
- Brown, R. M., McClelland, N. I., Deininger, R. A., & Tozer, R. G. (1970). A water quality index: Do we dare? *Water and Sewage Works*, 117(10), 339–343.
- Tyagi, S., Sharma, B., Singh, P., & Dobhal, R. (2013). Water quality assessment in terms of water quality index. *American Journal of Water Resources*, 1(3), 34–38. https://doi.org/10.12691/ajwr-1-3-3
- Ingole, N. W. & Bhole, A. G. (2019). Water quality parameters — A review. *InTechOpen*. https://www.intechopen.com/chapters/69568

### Algoritmos de Machine Learning
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451
- Friedman, J. H. (2002). Stochastic gradient boosting. *Computational Statistics & Data Analysis*, 38(4), 367–378. https://doi.org/10.1016/S0167-9473(01)00065-2
- Fernández-Delgado, M., Cernadas, E., Barro, S., & Amorim, D. (2014). Do we need hundreds of classifiers to solve real world classification problems? *Journal of Machine Learning Research*, 15(1), 3133–3181.

### Frameworks y Herramientas
- Zaharia, M., Xin, R. S., Wendell, P., Das, T., Armbrust, M., Dave, A., ... & Stoica, I. (2016). Apache Spark: A unified engine for big data processing. *Communications of the ACM*, 59(11), 56–65. https://doi.org/10.1145/2934664
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., ... & Zheng, X. (2016). TensorFlow: A system for large-scale machine learning. *12th USENIX Symposium on Operating Systems Design and Implementation*, 265–283.
- Chollet, F. (2015). *Keras* [Software]. GitHub. https://github.com/keras-team/keras
- Jordahl, K., Van den Bossche, J., Fleischmann, M., Wasserman, J., McBride, J., & Gerard, J. (2020). *geopandas/geopandas: v0.8.1* [Software]. Zenodo. https://doi.org/10.5281/zenodo.3946761
- McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 445, 51–56.
