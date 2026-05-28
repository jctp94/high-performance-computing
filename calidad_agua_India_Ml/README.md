# Análisis de Calidad del Agua con Machine Learning
### Ríos de la India — Implementación con PySpark y Scikit-Learn

**Proyecto II — Computación de Alto Rendimiento**
Juan Camilo Torres Peña · Maestría en Computación de Alto Rendimiento · 2026

---

## Descripción

Pipeline completo de análisis de calidad del agua en ríos de la India que combina procesamiento distribuido con Apache Spark + Hadoop HDFS y tres modelos de Machine Learning: una Red Neuronal para regresión continua del WQI y dos clasificadores de ensemble (Random Forest y Gradient Boosting) para predicción operacional directa desde mediciones crudas de sensores.

El dataset proviene del **Central Pollution Control Board (CPCB) de India** con 534 estaciones de monitoreo activas y 6 parámetros fisicoquímicos: Oxígeno Disuelto, pH, Conductividad, DBO, Nitratos/Nitritos y Coliformes Fecales.

---

## Estructura del repositorio

```
.
├── proyectoFinalImplementacion.ipynb   # Notebook principal — pipeline completo
├── proyectoFinalImplementacion.pdf     # PDF exportado del notebook con outputs
├── waterquality.csv                    # Dataset CPCB India (534 estaciones)
├── Indian_States/                      # Shapefile para visualización geoespacial
│   ├── Indian_States.shp
│   ├── Indian_States.dbf
│   ├── Indian_States.prj
│   └── Indian_States.shx
├── CADProyecto.pptx                    # Presentación del proyecto (19 diapositivas)
├── diapositivas_presentacion.md        # Presentación en formato Marp (20 diapositivas)
├── script_video_presentacion.md        # Guion para video de presentación (~15 min)
├── informe_modelo1_random_forest.md    # Informe técnico — Random Forest
├── informe_modelo2_gradient_boosting.md # Informe técnico — Gradient Boosting
├── High-Performance-Computing.pdf      # Documento de referencia del curso
└── README.md
```

---

## Pipeline

```
HDFS (waterquality.csv)
        │
        ▼
SparkSession — Carga distribuida
        │
        ▼
ETL con PySpark SQL
  · Filtro de nulos · Cast de tipos · Vistas temporales
        │
        ▼
Cálculo de sub-índices qr* (UDFs Spark)
  qrPH · qrDO · qrCOND · qrBOD · qrNN · qrFecal
        │
        ▼
WQI = Σ(pesoᵢ × qrᵢ)   →   CALIDAD = f(WQI)
        │
        ▼
Visualización geoespacial (GeoPandas + shapefile India)
        │
        ▼
.toPandas() → nodo driver
        │
        ├── Modelo 1: ANN (Keras)         — Regresión WQI continuo
        ├── Modelo 2: Random Forest        — Clasificación directa
        └── Modelo 3: Gradient Boosting    — Clasificación directa
```

---

## Modelos

### Modelo 1 — Red Neuronal Artificial (Keras / TensorFlow)

Tarea de **regresión**: predice el valor continuo del WQI a partir de los seis sub-índices calculados.

| Componente | Detalle |
|---|---|
| Arquitectura | Dense(350, ReLU) × 3 capas ocultas + Dense(1, lineal) |
| Optimizador | Adam — lr=0.001 |
| Épocas | 200 · Batch: 16 |
| Preprocesamiento | StandardScaler |

**Resultados en test set (107 muestras):**

| Métrica | Valor |
|---|---|
| R² | **0.9989** |
| RMSE | 0.5137 |
| MAE | 0.1063 |
| Sesgo | 0.0381 |

> Las features de entrada (qr\*) son sub-índices derivados con las mismas reglas que producen el WQI, por lo que el R² cercano a 1 es matemáticamente esperado. El modelo no puede usarse directamente en campo donde solo existen mediciones crudas.

---

### Modelo 2 — Random Forest Classifier (Scikit-Learn)

Tarea de **clasificación multiclase**: predice directamente la categoría de calidad del agua (Excelente / Buena / Baja / Muy Baja / Inadecuada) a partir de las mediciones crudas.

| Hiperparámetro | Valor | Justificación |
|---|---|---|
| `n_estimators` | 300 | Estabiliza el error del ensemble |
| `max_depth` | 12 | Árboles profundos — bagging controla overfitting |
| `min_samples_leaf` | 3 | Evita hojas con casos únicos |
| `max_features` | `'sqrt'` | Diversidad entre árboles (Breiman, 2001) |
| `class_weight` | `'balanced'` | Compensa desbalanceo de clases |
| `n_jobs` | `-1` | Paralelización total de núcleos |

**Resultados en test set:**

| Métrica | Valor |
|---|---|
| Accuracy | 94.4% |
| F1-Score (macro) | 71.9% |
| AUC-ROC (macro) | 97.8% |
| CV F1 media (5-fold) | **85.1%** |

**Importancia de features:** FECAL_COLIFORM (34.7%) · DO (27.3%) · CONDUCTIVITY (13.7%) · BOD · NITRATE · pH

---

### Modelo 3 — Gradient Boosting Classifier (Scikit-Learn)

Tarea de **clasificación multiclase**: mismo objetivo que Random Forest pero usando boosting secuencial con early stopping.

| Hiperparámetro | Valor | Justificación |
|---|---|---|
| `n_estimators` | 300 | Límite superior — early stopping actúa antes |
| `learning_rate` | 0.05 | Shrinkage: lr bajo + muchos árboles = robusto |
| `max_depth` | 4 | Weak learners — GB amplifica errores complejos |
| `min_samples_leaf` | 5 | Más conservador que RF ante overfitting |
| `subsample` | 0.8 | Stochastic GB — 20% descartado por árbol |
| `validation_fraction` | 0.10 | Set interno para early stopping |
| `n_iter_no_change` | 25 | Para si no mejora en 25 rondas |

**Resultados en test set:**

| Métrica | Valor |
|---|---|
| Accuracy | 90.0% |
| F1-Score (macro) | 68.0% |
| AUC-ROC (macro) | 98.1% |
| CV F1 media (5-fold) | **87.9%** |
| Early stopping | Ronda 152 de 300 |

> GB obtiene mejor CV F1 (87.9% vs 85.1%) y mejor AUC-ROC que RF, lo que lo hace el modelo recomendado para producción.

---

## Categorías de calidad del agua (WQI)

| Rango WQI | Categoría | Acción recomendada |
|---|---|---|
| [0, 25) | Excelente | Apta sin tratamiento |
| [25, 50) | Buena | Tratamiento básico |
| [50, 75) | Baja | Tratamiento estándar |
| [75, 100) | Muy Baja | Tratamiento intensivo |
| ≥ 100 | Inadecuada | No apta — alerta crítica |

---

## Requisitos

### Entorno Hadoop / Spark (ETL)

El notebook está configurado para ejecutarse sobre un clúster con:

- Apache Spark (probado con Spark 3.x)
- Hadoop HDFS — namenode en `hdfs://10.195.34.34:9000`
- Dataset en HDFS: `/csv/waterquality.csv`

Para ejecución local sin clúster, reemplazar la carga desde HDFS por:

```python
df00 = sparkS.read.format("csv").option("header","true").load("waterquality.csv")
```

### Dependencias Python

```
pyspark
findspark
tensorflow / keras
scikit-learn
pandas
numpy
matplotlib
seaborn
geopandas
jupyter
nbconvert[webpdf]   # para exportar PDF
```

Instalación:

```bash
pip install pyspark findspark tensorflow scikit-learn pandas numpy \
            matplotlib seaborn geopandas jupyter
```

### Exportar notebook a PDF

```bash
jupyter nbconvert --to webpdf --no-input proyectoFinalImplementacion.ipynb
```

> Requiere Chromium/Playwright instalado. Alternativa: abrir `proyectoFinalImplementacion.html` en un navegador e imprimir a PDF.

---

## Ejecución

1. Levantar la sesión de Spark con la configuración del clúster (ver celda 3 del notebook).
2. Cargar el dataset desde HDFS (celda 5).
3. Ejecutar el pipeline ETL completo (celdas 7–62).
4. Generar la visualización geoespacial (celdas 63–79).
5. Entrenar la Red Neuronal (celdas 80–94).
6. Ejecutar los modelos de clasificación adicionales (celdas 95–126).

---

## Resultados comparativos

| Modelo | Tarea | Features | CV F1 | AUC-ROC | Uso recomendado |
|---|---|---|---|---|---|
| ANN (Keras) | Regresión WQI | Sub-índices qr\* | — | — | Análisis de tendencias |
| Random Forest | Clasificación | Mediciones crudas | 85.1% | 97.8% | Baseline rápido |
| Gradient Boosting | Clasificación | Mediciones crudas | **87.9%** | **98.1%** | **Producción** |

---

## Documentación adicional

- `informe_modelo1_random_forest.md` — Explicación detallada del algoritmo Random Forest, justificación de hiperparámetros y análisis de métricas.
- `informe_modelo2_gradient_boosting.md` — Explicación del algoritmo Gradient Boosting, fundamento matemático, early stopping y comparación con RF.
- `diapositivas_presentacion.md` — Presentación completa en formato Marp (20 diapositivas). Renderizable con [Marp CLI](https://github.com/marp-team/marp-cli) o la extensión de VS Code.
- `script_video_presentacion.md` — Guion de ~15 minutos para video de presentación, alineado a las 19 diapositivas del `CADProyecto.pptx` con métricas reales.

---

## Referencias

- Central Pollution Control Board — CPCB. (2022). *National Water Quality Monitoring Programme*. Government of India. https://cpcb.nic.in/
- Brown, R. M., McClelland, N. I., Deininger, R. A., & Tozer, R. G. (1970). A water quality index: Do we dare? *Water and Sewage Works*, 117(10), 339–343.
- Tyagi, S., Sharma, B., Singh, P., & Dobhal, R. (2013). Water quality assessment in terms of water quality index. *American Journal of Water Resources*, 1(3), 34–38. https://doi.org/10.12691/ajwr-1-3-3
- Ingole, N. W. & Bhole, A. G. (2019). Water quality parameters — A review. *InTechOpen*. https://www.intechopen.com/chapters/69568
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451
- Friedman, J. H. (2002). Stochastic gradient boosting. *Computational Statistics & Data Analysis*, 38(4), 367–378. https://doi.org/10.1016/S0167-9473(01)00065-2
- Fernández-Delgado, M., Cernadas, E., Barro, S., & Amorim, D. (2014). Do we need hundreds of classifiers to solve real world classification problems? *Journal of Machine Learning Research*, 15(1), 3133–3181.
- Zaharia, M., Xin, R. S., Wendell, P., Das, T., Armbrust, M., Dave, A., ... & Stoica, I. (2016). Apache Spark: A unified engine for big data processing. *Communications of the ACM*, 59(11), 56–65. https://doi.org/10.1145/2934664
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., ... & Zheng, X. (2016). TensorFlow: A system for large-scale machine learning. *12th USENIX Symposium on Operating Systems Design and Implementation*, 265–283.
- Chollet, F. (2015). *Keras* [Software]. GitHub. https://github.com/keras-team/keras
- Jordahl, K., Van den Bossche, J., Fleischmann, M., Wasserman, J., McBride, J., & Gerard, J. (2020). *geopandas/geopandas: v0.8.1* [Software]. Zenodo. https://doi.org/10.5281/zenodo.3946761
