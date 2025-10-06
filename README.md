# Proyecto ML inicial — clasificación y regresión

# A-Clasificación




## 1. Carga y Exploración Inicial de Datos
El proceso de EDA y limpieza fue desarrollado en el _notebook_ boston_eda1.ipynb.

- **Carga del Dataset**: Se utilizó la librería **pandas** para cargar los datos desde el archivo titanic.csv.
- **Análisis Preliminar**:
  - Se inspeccionaron las primeras filas del dataframe con df.head() para entender la estructura y el tipo de datos de cada columna.
  - Se utilizó df.info() para obtener un resumen de las columnas, sus tipos de datos y la cantidad de valores no nulos. Este paso fue crucial para identificar las columnas con datos faltantes.
  - Se confirmó con df.isnull().sum() la presencia de valores nulos en las columnas Age (177), Cabin (687) y Embarked (2).
  - Se verificó que no existían filas duplicadas en el conjunto de datos con df.duplicated().sum().

## 2\. Limpieza y Preprocesamiento de Datos

Esta fase se centró en manejar los datos faltantes y transformar las variables para que fueran utilizables por un modelo.

- **Tratamiento de Valores Nulos**:
  - **Age**: Los 177 valores nulos en la columna Age se imputaron utilizando la **mediana** de la edad de los pasajeros. Se eligió la mediana en lugar de la media para mitigar el efecto de los valores atípicos.
  - **Embarked**: Los 2 valores nulos en Embarked se rellenaron con la **moda** (el puerto de embarque más frecuente), que en este caso era 'S'.
  - **Cabin**: Debido a la gran cantidad de valores nulos (687 de 891), se decidió que la columna Cabin no aportaría información fiable y fue **eliminada** del dataset.
- **Eliminación de Columnas Irrelevantes**:
  - Se eliminaron las columnas PassengerId, Name, y Ticket, ya que son identificadores únicos o texto libre que no aportan valor predictivo generalizable para un modelo de clasificación.
- **Codificación de Variables Categóricas**:
  - **Sex**: La columna Sex se codificó utilizando LabelEncoder, convirtiendo 'male' y 'female' a valores numéricos (0 y 1).
  - **Embarked**: La columna Embarked se transformó mediante **One-Hot Encoding** utilizando pd.get_dummies. Se creó una nueva columna binaria para cada puerto de embarque, eliminando la primera categoría (drop_first=True) para evitar multicolinealidad.

## 3\. Visualización de Datos

Se generaron visualizaciones para entender mejor la distribución de los datos y la presencia de outliers.

- **Análisis de Distribución**: Se crearon **histogramas** para cada variable numérica, lo que permitió observar la forma de su distribución (e.g., la distribución de Age después de la imputación).
- **Detección de Outliers**: Se utilizaron **gráficos de caja (boxplots)** para las variables numéricas como Pclass, Age, SibSp, Parch, y Fare. Esto ayudó a identificar valores atípicos, especialmente en Fare. Para este análisis inicial, no se eliminaron outliers, pero su identificación es importante para futuras iteraciones del modelo.

## 4\. Resultado EDA

El proceso resultó en un **DataFrame limpio y preprocesado**, que fue exportado al archivo titanic_eda_final.csv.

## 5\. Modelos de Clasificación Evaluados

Se entrenaron y evaluaron los siguientes tres modelos:

- **Regresión Logística (LogisticRegression):** Un modelo lineal simple ideal para clasificación binaria, que estima la probabilidad de que un evento ocurra.
- **Árbol de Decisión (DecisionTreeClassifier):** Un modelo no paramétrico que toma decisiones secuenciales basadas en las características.
- **K-Nearest Neighbors (KNeighborsClassifier):** Un clasificador basado en la distancia que predice la clase de un punto de datos comparándolo con sus K vecinos más cercanos.

## 6\. Proceso de Modelado

**Preparación de Datos**

- **Carga:** Se cargó el _dataset_ limpio del Titanic (titanic_eda_cleaned.csv).
- **Escalado:** Se aplicó un escalador estándar (StandardScaler) a las características numéricas para asegurar que todas contribuyan de manera equitativa al entrenamiento de los modelos, un paso crucial especialmente para el modelo KNN.
- **División Entrenamiento/Prueba:** Los datos se dividieron en conjuntos de entrenamiento y prueba para evaluar el rendimiento del modelo en datos no vistos, evitando el sobreajuste.

**Entrenamiento y Evaluación**

Cada modelo fue entrenado en los datos de entrenamiento y evaluado utilizando las siguientes métricas:

- **Accuracy (Precisión o Exactitud):** La proporción de predicciones correctas sobre el total de casos.
- **Matriz de Confusión:** Tabla que permite visualizar el rendimiento del modelo, mostrando la cantidad de Verdaderos Positivos (VP), Verdaderos Negativos (VN), Falsos Positivos (FP) y Falsos Negativos (FN).
- **Reporte de Clasificación:** Incluye métricas detalladas como _Precision_, _Recall_ (Sensibilidad) y _F1-Score_ para cada clase.

## 7\. Resultados y Comparación de Modelos

**Comparación de Resultados**

Para evaluar y comparar el rendimiento de cada modelo, se utilizaron las siguientes métricas clave sobre el conjunto de datos de prueba.

| Modelo | Accuracy | Precision (Sobrevive) | Recall (Sobrevive) | F1-Score (Sobrevive) |
| --- | --- | --- | --- | --- |
| **Regresión Logística** | **0.8101** | **0.7857** | 0.7432 | **0.7639** |
| **Árbol de Decisión** | 0.7877 | 0.7368 | **0.7568** | 0.7467 |
| **k-Nearest Neighbors (k-NN)** | 0.8045 | 0.7826 | 0.7297 | 0.7552 |

**Conclusión de Resultados**

Basado en la tabla comparativa:

- La **Regresión Logística** obtiene el Accuracy (exactitud general) más alto, prediciendo correctamente el 81% de los casos.
- El **Árbol de Decisión** muestra el Recall más alto para la clase "Sobrevive", lo que significa que es el mejor modelo para identificar a los pasajeros que realmente sobrevivieron.
- La **Regresión Logística** y **k-NN** tienen una Precision muy similar, indicando que cuando predicen que alguien sobrevivió, aciertan en aproximadamente el 78% de las ocasiones.


**Modelo Recomendado:** La **Regresión Logística** es el más recomendable para este problema. Dado su **Mejor Rendimiento General:** Con un **Accuracy de 0.8101** y el **F1-Score más alto (0.7639)**, demuestra ser el modelo más equilibrado y con la mayor capacidad predictiva general.

Aunque el Árbol de Decisión tiene el recall más alto para los sobrevivientes, su accuracy y precision generales son inferiores, lo que lo hace menos robusto. El modelo k-NN es una alternativa muy competente, pero la Regresión Logística lo supera ligeramente en las métricas más importantes.

# B- Regresión



## 1. Carga y Exploración Inicial de Datos
El proceso de EDA y limpieza fue desarrollado en el _notebook_ boston_eda1.ipynb.
- **Carga del Dataset**: Se utilizó la librería **pandas** para cargar los datos desde el archivo boston.csv.
- **Análisis Preliminar**:
  - Se revisaron las primeras filas con df.head() para tener una visión general de la estructura.
  - Con df.info(), se identificó que varias columnas numéricas como CRIM, LON y LAT estaban incorrectamente cargadas como tipo object.
  - Se verificó con df.isnull().sum() y df.duplicated().sum() que, inicialmente, no existían valores nulos ni filas duplicadas aparentes.

## 2\. Limpieza y Preprocesamiento de Datos

El objetivo de esta fase fue corregir las inconsistencias detectadas.

- **Corrección de Tipos de Dato**:
  - Se convirtieron las columnas de tipo object que debían ser numéricas (CRIM, INDUS, etc.) a formato numérico utilizando pd.to_numeric. Se usó el parámetro errors='coerce' para convertir automáticamente en NaN (Not a Number) cualquier valor que no pudiera ser transformado.
  - Este proceso reveló valores no numéricos en la columna CRIM, generando valores nulos que debían ser tratados.
- **Eliminación de Columnas Irrelevantes**:
  - Se eliminaron las columnas OBS., TOWN, TRACT, LON y LAT, ya que son identificadores o datos geográficos que no aportan valor predictivo directo para un modelo de regresión general.
- **Tratamiento de Valores Nulos**:
  - Tras la conversión de tipos, se encontraron valores NaN únicamente en la columna CRIM.
  - Se decidió imputar estos valores nulos utilizando la **media** de la propia columna CRIM, una estrategia común para no perder registros.

## 3\. Visualización y Detección de Outliers

Para asegurar la robustez del futuro modelo, se realizó un análisis de valores atípicos.

- **Análisis de Distribución**: Se generaron **histogramas** para visualizar la distribución de cada variable numérica.
- **Detección de Outliers**:
  - Se utilizaron **gráficos de caja (boxplots)** para identificar visualmente la presencia de outliers en las variables.
  - Se calculó el porcentaje de outliers para cada columna mediante el **método del Rango Intercuartílico (IQR)**. Se observó un alto porcentaje de outliers en columnas como CRIM (14.90%), ZN (13.27%) y B (15.92%). La columna CHAS aparece con un 100% de outliers debido a su naturaleza binaria (0 o 1), donde el valor 1 es poco frecuente.
  - **Manejo de Outliers en la Variable Objetivo**: Se identificó que la variable objetivo MEDV (valor mediano de la vivienda) presentaba valores atípicos, especialmente un grupo de registros con valor 50.0. Estos valores suelen considerarse "censurados" o topados en este dataset. Para evitar que sesguen el modelo, se decidió **eliminar todos los registros donde MEDV fuera igual o superior a 50.0**.

## 4\. Resultados

El resultado principal de este proceso es un **DataFrame limpio y preprocesado**, guardado en el archivo boston_eda_final.csv.

## 5\. Modelos de Regresión Evaluados

Se entrenaron y evaluaron tres modelos distintos para este problema:

- **Regresión Lineal (LinearRegression):**
- **Árbol de Regresión (DecisionTreeRegressor):**
- **Random Forest Regressor (RandomForestRegressor**

## 6\. Proceso de Modelado.

**Preparación de Datos.**

- **División de Características y Objetivo:** Se definieron las características de entrada (X) y la variable objetivo (Y = MEDV).
- **División Entrenamiento/Prueba:** El conjunto de datos se dividió en subconjuntos de entrenamiento y prueba (X_train, X_test, Y_train, Y_test) para asegurar que la evaluación del modelo se realice sobre datos no vistos.

**Entrenamiento y Evaluación**

Para cada uno de los tres modelos, se siguió el siguiente procedimiento:

- **Entrenamiento:** El modelo se ajustó a los datos de entrenamiento (X_train, Y_train).
- **Predicción:** Se generaron predicciones sobre el conjunto de prueba (X_test).
- **Métricas:** Se calcularon las siguientes métricas de rendimiento:
  - **RMSE (Root Mean Squared Error):** Error cuadrático medio de la raíz (medida de la magnitud de los errores).
  - **MAE (Mean Absolute Error):** Error absoluto medio (medida de la desviación promedio).
  - R2 **Score (Coeficiente de Determinación):** Indica la proporción de la varianza en la variable dependiente que es predecible a partir de las variables independientes.

## 7\. Resultados y Comparación de Modelos

**Comparación de Resultados**

Los resultados obtenidos al evaluar los tres modelos en el conjunto de prueba se resumen a continuación:

| **Modelo** | **RMSE (Error Cuadrático Medio)** | **MAE (Error Absoluto Medio)** | **R2 (Coeficiente de Determinación)** |
| --- | --- | --- | --- |
| **LinearRegression** | **0.397536** | **0.145486** | **0.996906** |
| DecisionTreeRegressor | 0.984419 | 0.290816 | 0.981030 |
| RandomForestRegressor | 0.534418 | 0.193367 | 0.994409 |

**Conclusión de Resultados**

Basado en la tabla comparativa:

- El modelo de **Regresión Lineal** fue el que mostró el mejor rendimiento, logrando el valor de R2 más alto (**0.9969**) y los errores (RMSE y MAE) más bajos. Un R2 tan cercano a 1 indica que el modelo es capaz de explicar casi toda la varianza en la variable objetivo con las características dadas.
- El **Random Forest Regressor** se desempeñó como el segundo mejor modelo, con un R2 de **0.9944**, también un rendimiento muy sólido.
- El **Decision Tree Regressor** fue el modelo con el rendimiento más bajo, con errores más altos y el R2 más bajo de **0.9810**.

**Modelo Recomendado:** La **Regresión Lineal** es la opción más efectiva y simple para este conjunto de datos, dada su excelente capacidad predictiva y menor complejidad computacional en comparación con los modelos de _ensemble_ (Random Forest).



