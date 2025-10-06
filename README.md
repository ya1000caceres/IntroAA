# Proyecto ML inicial — clasificación y regresión

### A-Clasificación
**1. Análisis Exploratorio de Datos (EDA) y Preprocesamiento**

El proceso de EDA y limpieza fue desarrollado en el *notebook*
classification_eda1.ipynb.

**1.1. Gestión de Valores Nulos**

Se identificaron y trataron valores nulos en el conjunto de datos de la
siguiente manera:

- **Age (Edad):** Los 177 valores nulos se imputaron utilizando la
  **mediana** de la columna.

- **Embarked (Puerto de Embarque):** Los 2 valores nulos se imputaron
  con la **moda** de la columna (el valor más frecuente).

- **Cabin (Cabina):** La columna se **eliminó** del conjunto de datos
  debido a la gran cantidad de valores faltantes (687 de 891).

**1.2. Eliminación y Codificación de Variables**

Se realizaron transformaciones clave en las características (features)
para hacerlas aptas para el modelado:

- **Eliminación de Columnas:** Se eliminaron las columnas PassengerId,
  Name y Ticket por ser identificadores o texto con baja predictibilidad
  directa.

- **Codificación de Sex:** La variable binaria Sex (male/female) se
  transformó a una variable numérica Sex_Encoded utilizando
  **LabelEncoder**.

- **Codificación de Embarked:** La variable nominal Embarked se
  transformó utilizando **One-Hot Encoding** (pd.get_dummies) con
  eliminación de la primera columna para evitar multicolinealidad
  (drop_first=True).

El conjunto de datos limpio se exportó a un archivo CSV
(titanic_eda_final.csv) para su uso posterior en la fase de modelado.

**2. Modelado y Entrenamiento**

El proceso de entrenamiento de los modelos se documenta en el *notebook*
classification.ipynb.

**2.1. Preparación de Datos**

1.  **Definición de Variables:**

    - **Variable Objetivo (Y):** Survived.

    - **Variables Predictoras (X):** El resto de las columnas.

2.  **División de Datos:** El conjunto de datos se dividió en conjuntos
    de entrenamiento y prueba (Train/Test) en una proporción **80/20**
    (test_size=0.2).

3.  **Escalado de Características:** Las variables numéricas en los
    conjuntos de entrenamiento y prueba se escalaron utilizando
    **StandardScaler** para estandarizar los datos

**2.2. Modelos Entrenados**

Se entrenaron y evaluaron tres modelos de clasificación diferentes para
comparar su rendimiento:

1.  **Regresión Logística** (LogisticRegression)

2.  **Árbol de Decisión** (DecisionTreeClassifier)

3.  **K-Vecinos Más Cercanos** (KNeighborsClassifier)

## 3 Evaluación de Resultados 

A continuación, se presenta evaluación de los modelos

**3.1. Regresión Logística**

- **Matriz de Confusión**: El modelo predijo correctamente la no
  supervivencia de 90 pasajeros (TN) y la supervivencia de 55 pasajeros
  (TP). Cometió 15 falsos positivos (FP) y 19 falsos negativos (FN).

- **Reporte de Clasificación (Clase \"Sobrevive\" / 1)**: Con una
  **Precision de 0.79**, el 79% de las predicciones de \"Sobrevive\"
  fueron correctas. Con un **Recall de 0.74**, capturó al 74% de los
  pasajeros que realmente sobrevivieron.

**3.2. Árbol de Decisión**

- **Matriz de Confusión**: El modelo tuvo la mayor cantidad de **Falsos
  Negativos (24)**, es decir, predijo que 24 personas no sobrevivirían,
  cuando en realidad sí lo hicieron.

- **Reporte de Clasificación (Clase \"Sobrevive\" / 1)**: Mostró el
  **Recall más bajo (0.68)**, lo que indica que fue el menos efectivo
  para identificar a los supervivientes reales.

**3.3. K-Vecinos Más Cercanos (KNN)**

- **Matriz de Confusión**: Este modelo fue el más equilibrado y el que
  mejor rendimiento tuvo en general. Tuvo el mayor número de
  predicciones correctas de no supervivencia (TN=92) y supervivencia
  (TP=56).

- **Reporte de Clasificación (Clase \"Sobrevive\" / 1)**: Consiguió la
  **Precision (0.81)** y el **Recall (0.76)** más altos para la clase de
  supervivencia, lo que se traduce en el mejor **F1-Score (0.78)** y
  confirma su superioridad en este ejercicio de clasificación.

#### **4. Análisis Detallado por Modelo:** 
**4.1. Regresión Logística**

- **Matriz de Confusión**: El modelo predijo correctamente la no
  supervivencia de 90 pasajeros (TN) y la supervivencia de 55 pasajeros
  (TP). Cometió 15 falsos positivos (FP) y 19 falsos negativos (FN).

- **Reporte de Clasificación (Clase \"Sobrevive\" / 1)**: Con una
  **Precision de 0.79**, el 79% de las predicciones de \"Sobrevive\"
  fueron correctas. Con un **Recall de 0.74**, capturó al 74% de los
  pasajeros que realmente sobrevivieron.

**4.2. Árbol de Decisión**

- **Matriz de Confusión**: El modelo tuvo la mayor cantidad de **Falsos
  Negativos (24)**, predijo que 24 personas no sobrevivirían, cuando en
  realidad sí lo hicieron.

- **Reporte de Clasificación (Clase \"Sobrevive\" / 1)**: Mostró el
  **Recall más bajo (0.68)**, lo que indica que fue el menos efectivo
  para identificar a los supervivientes reales.

**4.3. K-Vecinos Más Cercanos (KNN)**

- **Matriz de Confusión**: Este modelo fue el más equilibrado y el que
  mejor rendimiento tuvo en general. Tuvo el mayor número de
  predicciones correctas de no supervivencia (TN=92) y supervivencia
  (TP=56).

- **Reporte de Clasificación (Clase \"Sobrevive\" / 1)**: Consiguió la
  **Precision (0.81)** y el **Recall (0.76)** más altos para la clase de
  supervivencia, lo que se traduce en el mejor **F1-Score (0.78)** y
  confirma su superioridad en este ejercicio de clasificación.

## 5. Conclusión General  

Basado en las métricas de evaluación detalladas, el modelo **K-Vecinos
Más Cercanos (KNN)** es inequívocamente el que ofrece el mejor
rendimiento general para predecir la supervivencia en este problema de
clasificación.

El alto valor de *Accuracy* del 82.68% es un excelente indicador de su
rendimiento global. Sin embargo, su verdadera fortaleza reside en el
balance que logra entre **Precision** y **Recall** (demostrado por el
F1-Score de 0.78). En un contexto como este, es deseable un *Recall*
alto para minimizar la subestimación de la supervivencia real (reducir
los Falsos Negativos), y el KNN lo maneja mejor que el Árbol de Decisión
o la Regresión Logística.



