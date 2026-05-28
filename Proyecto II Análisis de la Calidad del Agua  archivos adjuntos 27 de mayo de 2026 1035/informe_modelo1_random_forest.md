# Informe Técnico — Modelo 1: Random Forest Classifier
### Predicción de Calidad del Agua — Ríos de la India
**Maestría en Computación de Alto Rendimiento**

---

## 1. ¿Qué problema resuelve este modelo?

El modelo clasifica cada muestra de agua en una de cinco categorías de calidad:

| Categoría | Significado operacional |
|---|---|
| `Excelente` | Agua dulce, apta para consumo sin tratamiento |
| `Buena` | Contaminación leve, requiere tratamiento básico |
| `Baja` | Contaminación moderada |
| `Muy_Baja` | Contaminación severa |
| `Inadecuada` | Agua residual, no apta para ningún uso doméstico |

**Entradas del modelo** (parámetros fisicoquímicos directos):
- DO, pH, CONDUCTIVITY, BOD, NITRATE_N_NITRITE_N, FECAL_COLIFORM

**Diferencia clave con la Red Neuronal original:** la ANN predice el valor numérico continuo del WQI usando sub-índices ya calculados. Este modelo predice directamente la categoría de acción a partir de las mediciones crudas — sin cómputo intermedio.

---

## 2. ¿Cómo funciona Random Forest? (Intuición completa)

### 2.1 El problema de los árboles de decisión individuales

Un árbol de decisión individual aprende reglas del tipo:
```
Si DO < 4.1  Y  BOD > 6.0  →  Muy_Baja
```
El problema es que un árbol entrenado sin restricciones se **sobreajusta** (overfitting): memoriza el conjunto de entrenamiento pero falla al generalizar.

### 2.2 La solución: diversidad forzada

Random Forest crea **B árboles independientes entre sí**, introduciendo aleatoriedad en dos niveles:

**Nivel 1 — Bootstrap Sampling (Bagging):**
Cada árbol se entrena sobre una muestra aleatoria *con reemplazo* del dataset original. Si hay N muestras, cada árbol ve aproximadamente el 63% del conjunto (el 37% restante se llama *Out-of-Bag* y puede usarse para validación gratuita).

```
Dataset original:  [A, B, C, D, E, F]
Árbol 1 entrena:   [A, A, C, D, D, F]  ← algunas repetidas, otras ausentes
Árbol 2 entrena:   [B, B, C, C, E, F]
Árbol 3 entrena:   [A, B, D, E, E, F]
```

**Nivel 2 — Feature Bagging:**
En cada nodo de cada árbol, en lugar de evaluar todas las 6 features, se eligen **√6 ≈ 2-3 features al azar** para buscar el mejor corte. Esto evita que todas los árboles se parezcan entre sí (por ejemplo, que todos usen FECAL_COLIFORM como primera división).

### 2.3 La predicción final: votación democrática

Una muestra nueva pasa por los B árboles. Cada árbol emite un voto:
```
Árbol 1: "Buena"
Árbol 2: "Buena"
Árbol 3: "Baja"
...
Árbol 300: "Buena"

Resultado final: "Buena"  (mayoría de votos)
```

La clave matemática es la **Ley de los Grandes Números**: si cada árbol tiene una tasa de error ε < 0.5, y los árboles son suficientemente independientes, el error del ensemble decrece exponencialmente con B.

---

## 3. Justificación de la elección del modelo

| Criterio del dataset | Por qué favorece RF |
|---|---|
| **Datos tabulares heterogéneos** | RF no asume distribución de los datos; trabaja igual con variables en distintas escalas (DO: 0-12 mg/L vs. FC: 0-300,000 UFC) |
| **~400-500 muestras** | RF funciona bien en rangos medianos; no requiere miles de datos como las redes neuronales profundas |
| **Outliers confirmados** | Los cortes del árbol son por percentil, no por distancia — un valor de FC=300,000 no "distorsiona" el modelo |
| **Sin normalización necesaria** | Los árboles no usan distancias ni gradientes; ordenan los datos, no los escalan |
| **Desbalanceo de clases** | `class_weight='balanced'` penaliza más los errores en clases minoritarias sin necesidad de SMOTE |
| **Interpretabilidad requerida** | La importancia de features (Gini) explica directamente qué parámetros monitorear |

---

## 4. Hiperparámetros: qué es cada uno y por qué se eligió ese valor

### `n_estimators = 300`
**¿Qué hace?** Define cuántos árboles componen el ensemble.

**¿Por qué 300?**
- Con pocos árboles (< 50), el modelo es inestable — pequeños cambios en los datos producen predicciones distintas.
- A partir de ~200 árboles, el error se estabiliza (meseta de la curva de error vs. B).
- 300 da un margen de seguridad sin costo computacional excesivo en un dataset pequeño.
- Más árboles nunca empeoran el modelo (solo lo hacen más lento).

### `max_depth = 12`
**¿Qué hace?** Limita cuán profundo puede crecer cada árbol. Un árbol de profundidad D puede hacer hasta 2^D divisiones.

**¿Por qué 12?**
- Sin límite, cada árbol memoriza el set de entrenamiento (overfitting total).
- Con profundidad 12 se permiten reglas con hasta 12 condiciones encadenadas — suficiente para capturar la complejidad del problema de calidad del agua.
- Valores más pequeños (ej. 5) limitan la capacidad del modelo (underfitting).

### `min_samples_leaf = 3`
**¿Qué hace?** Exige que cada hoja terminal tenga al menos 3 muestras. Evita hojas con 1 sola muestra (memorización pura).

**¿Por qué 3?**
- Con dataset pequeño (~400 muestras), es importante evitar que los árboles creen reglas para casos únicos.
- Valor 1 = overfitting. Valor muy grande (ej. 20) = underfitting.
- 3 es un balance conservador para datasets de este tamaño.

### `max_features = 'sqrt'`
**¿Qué hace?** En cada nodo de decisión, sólo se evalúan √(n_features) = √6 ≈ 2 features aleatorias para encontrar el mejor corte.

**¿Por qué sqrt?**
- Es la recomendación estándar de la literatura para clasificación (Breiman, 2001).
- Fuerza diversidad entre árboles: si siempre se evalúan todas las features, los árboles se parecen demasiado y pierden el beneficio del ensemble.
- Para regresión se usa n_features/3 en cambio.

### `class_weight = 'balanced'`
**¿Qué hace?** Ajusta el peso de cada clase inversamente proporcional a su frecuencia:
```
peso_clase_i = N_total / (N_clases × N_muestras_clase_i)
```

**¿Por qué balanced?**
- Si `Inadecuada` tiene 20 muestras y `Buena` tiene 150, sin ajuste el modelo ignorará `Inadecuada` para maximizar accuracy.
- Con `balanced`, un error en `Inadecuada` penaliza ~7.5 veces más que un error en `Buena`.
- Esencial en este dominio: clasificar como `Buena` un agua `Inadecuada` tiene consecuencias graves para la salud pública.

### `random_state = 42`
**¿Qué hace?** Fija la semilla del generador de números aleatorios.

**¿Por qué?** Reproducibilidad. Garantiza que ejecutar el código dos veces produce exactamente el mismo modelo — requisito en investigación científica y presentaciones.

### `n_jobs = -1`
**¿Qué hace?** Usa todos los núcleos del procesador disponibles para entrenar los árboles en paralelo.

**¿Por qué?** Los 300 árboles son completamente independientes entre sí — paralelización perfecta (speedup lineal con el número de núcleos).

---

## 5. Preprocesamiento aplicado

### División Train/Test estratificada (80/20)
```python
train_test_split(X, y_enc, test_size=0.20, stratify=y_enc)
```
El parámetro `stratify=y_enc` garantiza que la proporción de cada clase sea idéntica en entrenamiento y prueba. Sin esto, por azar el test podría no tener muestras de `Inadecuada`, haciendo la evaluación inválida.

### LabelEncoder
Convierte las etiquetas string (`'Buena'`, `'Inadecuada'`, etc.) en enteros (0, 1, 2, 3, 4) que sklearn puede procesar. Se fijó el orden semántico sobre `ORDEN_CLASES` para consistencia entre celdas.

### Sin normalización
Random Forest no requiere escalado porque los árboles trabajan con **comparaciones de orden** (`feature_i < umbral`), no con distancias euclidianas. Escalar DO de mg/L a unidades adimensionales no cambia qué samples caen a cada lado del corte.

---

## 6. Métricas de evaluación: qué mide cada una

### Accuracy
```
Accuracy = (predicciones correctas) / (total de predicciones)
```
**Limitación:** con clases desbalanceadas puede ser engañosa. Un modelo que predice siempre "Buena" tendría accuracy alta pero sería inútil.

### Precision (por clase)
```
Precision_clase_i = TP_i / (TP_i + FP_i)
```
De todas las muestras que el modelo dijo que son clase i, ¿cuántas realmente lo son? Alta precision = pocas falsas alarmas.

### Recall (por clase)
```
Recall_clase_i = TP_i / (TP_i + FN_i)
```
De todas las muestras que realmente son clase i, ¿cuántas detectó el modelo? Alto recall = pocas omisiones.

### F1-Score (macro)
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Media armónica de Precision y Recall. El promedio **macro** trata todas las clases por igual — es la métrica principal cuando hay desbalanceo.

### AUC-ROC (macro One-vs-Rest)
Para cada clase i, construye un clasificador binario "¿es clase i o no?". El AUC mide el área bajo la curva ROC: 1.0 = clasificador perfecto, 0.5 = aleatorio. El promedio macro sobre todas las clases da la capacidad discriminativa global.

### Validación Cruzada Estratificada (5-fold)
Divide el dataset en 5 partes iguales con estratificación. Entrena 5 veces usando 4 partes y valida en la 5a. Reporta media ± desv. estándar del F1-macro. **Más confiable que un único split** cuando el dataset es pequeño (~400 muestras).

---

## 7. Importancia de features: interpretación

La importancia Gini mide cuánto reduce en promedio la impureza de los nodos cada feature, ponderado por el número de muestras que pasan por esos nodos. Un valor alto significa que esa variable discrimina mejor entre clases.

**Interpretación esperada para calidad del agua:**
- **FECAL_COLIFORM** y **BOD**: indicadores directos de contaminación — dominan la clasificación.
- **DO**: indicador de salud del ecosistema — alta importancia consistente.
- **pH**, **CONDUCTIVITY**, **NITRATE_N_NITRITE_N**: información complementaria, menor importancia relativa.

Las barras de error (desv. estándar entre árboles) miden la estabilidad de la importancia: barras pequeñas = el ranking es robusto.

---

## 8. Interpretación de la matriz de confusión

La **diagonal principal** representa predicciones correctas. Los valores fuera de la diagonal son errores.

**Errores esperados más frecuentes:**
- Confusión entre `Buena` y `Baja` (clases adyacentes con parámetros similares en el borde).
- Raramente confusión entre `Excelente` e `Inadecuada` (son extremos opuestos en el espacio de features).

La **versión normalizada** (% por fila) muestra el recall de cada clase: qué porcentaje de las muestras reales de cada categoría fue clasificado correctamente.

---

## 9. Ventajas y limitaciones

### Ventajas
- No requiere normalización de datos.
- Robusto a outliers y valores faltantes (con imputación previa).
- Alta interpretabilidad: feature importances orientan el monitoreo.
- Entrenamiento paralelizable — rápido con `n_jobs=-1`.
- Provee probabilidades calibradas por clase vía `predict_proba`.

### Limitaciones
- Consumo de memoria proporcional a `n_estimators × profundidad_árbol`.
- Menos preciso que Gradient Boosting en datasets tabulares pequeños según literatura.
- No extrapola bien fuera del rango de entrenamiento (inherente a todos los árboles).

---

*Implementación: scikit-learn `RandomForestClassifier` | Dataset: CPCB India Water Quality Monitoring*
