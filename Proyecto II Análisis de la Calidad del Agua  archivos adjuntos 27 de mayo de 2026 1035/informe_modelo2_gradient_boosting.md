# Informe Técnico — Modelo 2: Gradient Boosting Classifier
### Predicción de Calidad del Agua — Ríos de la India
**Maestría en Computación de Alto Rendimiento**

---

## 1. ¿Qué problema resuelve este modelo?

El modelo clasifica muestras de agua en cinco categorías de calidad:

| Categoría | Significado operacional |
|---|---|
| `Excelente` | Agua dulce, apta para consumo sin tratamiento |
| `Buena` | Contaminación leve, tratamiento básico |
| `Baja` | Contaminación moderada |
| `Muy_Baja` | Contaminación severa |
| `Inadecuada` | Agua residual, no apta para uso doméstico |

**Entradas del modelo:** DO, pH, CONDUCTIVITY, BOD, NITRATE_N_NITRITE_N, FECAL_COLIFORM

Gradient Boosting resuelve el mismo problema de clasificación que Random Forest pero desde una **filosofía completamente diferente**: en lugar de construir árboles independientes en paralelo, los construye **uno a uno, cada uno corrigiendo los errores del anterior**.

---

## 2. ¿Cómo funciona Gradient Boosting? (Intuición completa)

### 2.1 La idea central: aprender de los errores

Imagina que estás aprendiendo a tirar dardos. Después de cada tiro, mides exactamente cuánto fallaste y en qué dirección. El siguiente tiro lo ajustas para corregir ese error específico. Gradient Boosting hace exactamente eso, pero con árboles de decisión.

### 2.2 El proceso paso a paso

**Paso 0 — Predicción inicial (modelo constante):**
El ensemble comienza prediciendo la clase más frecuente o la probabilidad base de cada clase. Este es el "primer tiro al dardo".

**Paso 1 — Calcular los residuos:**
Compara las predicciones actuales con las etiquetas reales. Calcula el **gradiente de la función de pérdida** respecto a las predicciones — esencialmente "en qué muestras y en qué dirección está fallando el modelo actual".

**Paso 2 — Entrenar un árbol sobre los residuos:**
Se entrena un árbol de decisión **pequeño y superficial** (weak learner) para predecir esos residuos (errores). Este árbol no aprende el problema original, aprende *qué corregir*.

**Paso 3 — Actualizar el ensemble:**
```
nuevo_ensemble = ensemble_anterior + learning_rate × árbol_nuevo
```
El `learning_rate` controla cuánto se "mueve" el ensemble con cada corrección (como la longitud del paso en descenso de gradiente).

**Paso 4 — Repetir:**
Los pasos 1-3 se repiten hasta que se alcanza el número de estimadores configurado o se activa el early stopping.

### 2.3 Visualización del proceso

```
Árbol 1:  Predice bien los casos claros. 
          Error residual: confunde clases intermedias.

Árbol 2:  Se enfoca en corregir esos casos intermedios.
          Error residual: ahora falla en algunos casos extremos.

Árbol 3:  Corrige los extremos mal clasificados por árbol 2.
          ...

Árbol 300: El ensemble acumulado tiene errores muy pequeños.
```

**Clave conceptual:** mientras Random Forest promedia árboles independientes (reducción de varianza), Gradient Boosting reduce el **sesgo** del modelo de forma secuencial. Por eso GB funciona mejor cuando el problema tiene patrones complejos que un árbol simple no puede capturar.

### 2.4 La función de pérdida multiclase

Para clasificación con K clases, GB minimiza la **log-loss (cross-entropy)**:

```
L = -Σ y_k × log(p_k)
```

Donde y_k es 1 si la muestra pertenece a la clase k y p_k es la probabilidad predicha. Cada árbol nuevo empuja las probabilidades hacia la clase correcta.

---

## 3. Justificación de la elección del modelo

| Criterio del dataset | Por qué favorece GB |
|---|---|
| **Dataset tabular estructurado** | GB es el estado del arte en competencias de datos tabulares (Kaggle, UCI); supera sistemáticamente a RF y redes neuronales en este tipo de datos |
| **~400-500 muestras** | Con early stopping, GB evita overfitting automáticamente incluso con datasets pequeños |
| **Relaciones no lineales complejas** | El boosting secuencial captura interacciones entre variables que los árboles individuales no ven |
| **Desbalanceo de clases** | El gradiente puede ponderarse por clase, penalizando más los errores en clases minoritarias |
| **Outliers confirmados** | Igual que RF, usa árboles → insensible a valores extremos |
| **Importancia de features** | Provee ranking de importancias basado en reducción de pérdida (más robusto que Gini de RF) |

**Referencia bibliográfica:** Fernández-Delgado et al. (2014) evaluaron 179 clasificadores en 121 datasets y encontraron que Random Forest y Gradient Boosting son consistentemente los mejores para datos tabulares.

---

## 4. Hiperparámetros: qué es cada uno y por qué se eligió ese valor

### `n_estimators = 300`
**¿Qué hace?** Número máximo de árboles (etapas de boosting) que se entrenarán.

**¿Por qué 300?**
- GB se beneficia de más estimadores que RF porque cada uno hace una corrección pequeña.
- 300 da suficiente capacidad para un problema con 5 clases y 6 features.
- Con `early stopping` activo, el entrenamiento puede detenerse antes si el modelo ya convergió — 300 es el límite superior, no necesariamente el número usado.

**Diferencia con RF:** en RF, agregar más árboles siempre es seguro (sólo baja la varianza). En GB sin regularización, agregar más árboles puede causar overfitting — por eso se combina con `learning_rate` pequeño y `early stopping`.

### `learning_rate = 0.05`
**¿Qué hace?** También llamado *shrinkage*. Escala la contribución de cada árbol al ensemble:
```
ensemble_t = ensemble_{t-1} + 0.05 × árbol_t
```

**¿Por qué 0.05?**
- Un `learning_rate` alto (ej. 0.5) aprende rápido pero se sobreajusta.
- Un `learning_rate` bajo (ej. 0.01) es más robusto pero requiere más estimadores.
- 0.05 es el valor recomendado en la literatura para datasets pequeños: permite convergencia estable sin overfitting.
- **Relación inversa con n_estimators:** `learning_rate` pequeño + muchos estimadores = mejor modelo que `learning_rate` grande + pocos estimadores (con el mismo costo computacional).

### `max_depth = 4`
**¿Qué hace?** Profundidad máxima de **cada árbol individual** (weak learner).

**¿Por qué 4?**
- En GB los árboles deben ser *débiles* a propósito. Un árbol de profundidad 4 puede capturar interacciones de hasta 4 variables pero no memoriza el dataset.
- Profundidades recomendadas para GB: entre 3 y 6.
- Valores más altos hacen cada árbol más poderoso pero aumentan riesgo de overfitting.
- **Contraste con RF:** en RF se usan árboles profundos (max_depth=12) porque el bagging controla el overfitting. En GB los árboles deben ser superficiales porque el boosting amplifica sus errores si son muy complejos.

### `min_samples_leaf = 5`
**¿Qué hace?** Mínimo de muestras en cada hoja terminal.

**¿Por qué 5?**
- Previene que el modelo cree reglas para casos únicos.
- Con dataset pequeño, exigir mínimo 5 muestras por hoja da estimaciones de probabilidad más estables.
- Más conservador que en RF (3) porque GB es más propenso a overfitting.

### `subsample = 0.8`
**¿Qué hace?** En cada etapa de boosting, usa sólo el 80% de las muestras elegidas aleatoriamente (sin reemplazo).

**¿Por qué 0.8?**
- Esto convierte el algoritmo en **Stochastic Gradient Boosting** (Friedman, 1999).
- La submuestra introduce aleatoriedad similar al bagging de RF.
- Efectos: reduce overfitting, acelera el entrenamiento, y a veces mejora la generalización.
- 0.8 es un valor estándar: descarta el 20% de las muestras en cada árbol, suficiente para añadir varianza sin perder demasiada información.
- **Con subsample < 1.0**, cada árbol ve datos ligeramente diferentes — los árboles se vuelven menos correlacionados.

### `max_features = 'sqrt'`
**¿Qué hace?** Igual que en RF: en cada nodo de cada árbol, sólo se evalúan √6 ≈ 2 features al azar.

**¿Por qué sqrt?**
- Agrega una segunda fuente de aleatoriedad (además de `subsample`).
- Hace los árboles más diversos entre sí → mejor ensemble.
- Reduce el costo computacional de cada árbol.

### `validation_fraction = 0.10`
**¿Qué hace?** Reserva el 10% del conjunto de entrenamiento como set de validación interno para early stopping.

**¿Por qué 0.10?**
- Este 10% nunca se usa para entrenar los árboles — sólo para medir si el modelo sigue mejorando.
- Con dataset de ~400 muestras, 10% ≈ 40 muestras de validación interna — suficiente para detectar sobreajuste.
- Valores más grandes (ej. 0.2) dejan menos datos de entrenamiento real.

### `n_iter_no_change = 25`
**¿Qué hace?** Activa el **early stopping**: si la métrica de validación no mejora en 25 rondas consecutivas, el entrenamiento se detiene.

**¿Por qué 25?**
- Previene overfitting automáticamente.
- 25 rondas da margen suficiente para distinguir una meseta real de una fluctuación temporal.
- Si se usa learning_rate bajo (0.05), la mejora por ronda es pequeña — se necesitan más rondas de "no mejora" para confirmar convergencia.

### `tol = 1e-4`
**¿Qué hace?** Umbral mínimo de mejora para considerar que el modelo progresó en una ronda.

**¿Por qué 1e-4?**
- Mejoras menores a 0.0001 en el score de validación se consideran ruido estadístico.
- Evita continuar entrenando cuando el modelo ya alcanzó su límite de capacidad para estos datos.

### `random_state = 42`
Reproducibilidad. Fija las semillas del subsampling y del feature sampling en cada árbol.

---

## 5. Preprocesamiento aplicado

### StandardScaler
```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```
**Nota importante:** Gradient Boosting basado en árboles **no requiere escalado** (igual que RF). Sin embargo, se aplica por dos razones:
1. **Buenas prácticas**: estandarizar el pipeline facilita añadir otros modelos (SVM, regresión logística) en comparaciones futuras sin modificar el preprocesamiento.
2. **Reproducibilidad**: el scaler se ajusta (`fit`) sólo sobre el conjunto de entrenamiento y se aplica (`transform`) al test — nunca al revés, para evitar data leakage.

### División estratificada y LabelEncoder
Idéntico a Random Forest — ver informe Modelo 1.

---

## 6. Curva de Convergencia: cómo interpretarla

El atributo `train_score_` almacena la log-likelihood en el conjunto de entrenamiento tras cada árbol. La curva muestra:

```
Alta mejora inicial  →  El ensemble captura los patrones principales
Meseta posterior     →  Retornos marginales decrecientes
Early stopping       →  Se detiene cuando la validación no mejora
```

Si la curva cae abruptamente y luego se estabiliza: **convergencia saludable**.
Si la curva nunca se estabiliza: `n_estimators` puede ser insuficiente.
Si early stopping actúa muy pronto (ej. ronda 30): el `learning_rate` puede ser demasiado alto.

---

## 7. Métricas de evaluación

Mismas métricas que Random Forest (Accuracy, Precision, Recall, F1-macro, AUC-ROC macro, CV 5-fold). Revisar Sección 6 del informe de RF para las definiciones formales.

**Punto adicional específico de GB:**

### Curvas ROC por clase (One-vs-Rest)
Para cada clase i, se construye un problema binario: "¿es clase i o cualquier otra?". La curva ROC grafica:
- **Eje X (FPR):** Tasa de Falsos Positivos — ¿qué % de muestras que NO son clase i se clasifican como i?
- **Eje Y (TPR / Recall):** ¿qué % de muestras que SÍ son clase i se detectan correctamente?

**AUC = 1.0:** clasificador perfecto para esa clase.
**AUC = 0.5:** equivalente a tirar una moneda.

Las clases extremas (`Excelente`, `Inadecuada`) típicamente tienen AUC > 0.95 porque sus firmas fisicoquímicas son muy distintas. Las clases intermedias (`Baja`, `Muy_Baja`) tienen AUC más bajo por solapamiento en el espacio de features.

---

## 8. Ventajas y limitaciones de Gradient Boosting

### Ventajas
- **Mayor precisión** que RF en datos tabulares de tamaño mediano (evidencia empírica robusta).
- **Early stopping integrado**: previene overfitting automáticamente.
- **Importancias de features** basadas en reducción de pérdida (más informativas que Gini de RF).
- **Stochastic GB** (`subsample < 1`) añade robustez adicional.
- Maneja naturalmente multiclase (OvR interno) sin configuración adicional.

### Limitaciones
- **Entrenamiento secuencial**: no puede paralelizarse como RF (cada árbol depende del anterior).
- **Más sensible a hiperparámetros** que RF: un `learning_rate` mal elegido puede causar overfitting o underfitting severo.
- **Tiempo de inferencia** similar a RF (recorre todos los árboles), pero entrenamiento más lento.
- En datasets muy grandes (millones de muestras), alternativas como XGBoost o LightGBM son más eficientes.

---

## 9. Comparación directa GB vs. RF (por qué GB es el modelo recomendado)

| Aspecto | Random Forest | Gradient Boosting |
|---|---|---|
| Estrategia | Paralela (bagging) | Secuencial (boosting) |
| Tipo de error que reduce | Varianza | Sesgo |
| Sensibilidad a hiperparámetros | Baja | Media |
| Velocidad de entrenamiento | Rápida | Más lenta |
| Desempeño típico en tabular | Muy bueno | **Mejor** |
| Early stopping | No disponible | **Sí, integrado** |
| Mejor cuando... | Mucho ruido en datos | Patrones complejos a capturar |

**Recomendación final:** Gradient Boosting para entornos donde la precisión de clasificación es prioritaria (decisiones de salud pública, alertas de contaminación). Random Forest para despliegues donde velocidad de entrenamiento y simplicidad operacional son prioritarias.

---

## 10. Fundamento matemático resumido (para la exposición)

El ensemble en la etapa t se construye como:

```
F_t(x) = F_{t-1}(x) + η · h_t(x)
```

Donde:
- `F_t(x)` es el ensemble después del árbol t
- `η` es el `learning_rate`
- `h_t(x)` es el árbol entrenado sobre los **pseudo-residuos** (gradiente negativo de la pérdida)

Los pseudo-residuos para log-loss son:
```
r_i = y_i - p_{t-1}(x_i)
```
La diferencia entre la etiqueta real y la probabilidad predicha por el ensemble anterior. El árbol h_t aprende a mapear features → correcciones necesarias.

---

*Implementación: scikit-learn `GradientBoostingClassifier` | Dataset: CPCB India Water Quality Monitoring*
