"""
Construye el cuaderno del taller (notebooks/taller_analisis_predictivo.ipynb).
Permite al docente regenerar el cuaderno si modifica su contenido.
"""
import nbformat as nbf

celdas = []


def md(texto):
    celdas.append(nbf.v4.new_markdown_cell(texto.strip("\n")))


def code(texto):
    celdas.append(nbf.v4.new_code_cell(texto.strip("\n")))


md(r"""
# Taller guiado: Análisis predictivo a partir de un conjunto de datos

**Caso de estudio:** predicción del riesgo de deserción estudiantil.
**Herramienta:** Google Colab (Python 3).
**Duración estimada:** 2 horas.

## Presentación

En este taller los estudiantes llevarán a cabo un análisis predictivo completo sobre un conjunto de datos de 1.200 estudiantes universitarios. El objetivo es construir un modelo que estime si un estudiante desertará (`deserta = 1`) o permanecerá en la institución (`deserta = 0`).

## Objetivos de aprendizaje

Al finalizar, el estudiante estará en capacidad de:

1. Cargar y explorar un conjunto de datos en Google Colab.
2. Detectar y corregir problemas de calidad en los datos.
3. Preparar los datos y entrenar modelos de clasificación.
4. Evaluar e interpretar un modelo con métricas adecuadas.
5. Usar el modelo para predecir casos nuevos.

## Estructura del taller

| Sección | Contenido |
|---|---|
| 1 | Preparación del entorno y librerías |
| 2 | Carga y comprensión de los datos |
| 3 | Exploración inicial |
| 4 | Limpieza de los datos |
| 5 | Análisis exploratorio visual |
| 6 | Preparación para el modelado |
| 7 | Entrenamiento y comparación de modelos |
| 8 | Evaluación e interpretación |
| 9 | Predicción para un estudiante nuevo |
| 10 | Actividades de cierre |

## Cómo trabajar en Colab

- Guardar una copia propia con **Archivo > Guardar una copia en Drive**.
- Ejecutar cada celda con **Shift + Enter**, en orden, de arriba hacia abajo.
- Si el entorno se reinicia, usar **Entorno de ejecución > Ejecutar anteriores**.
- Los bloques titulados **Actividad del estudiante** deben responderse en una celda de texto nueva (**Insertar > Celda de texto**).
""")

# 1. LIBRERÍAS
md(r"""
## 1. Preparación del entorno y librerías

Una **librería** es un conjunto de funciones ya programadas que resuelven tareas comunes. Todas las librerías de este taller vienen instaladas en Google Colab.

| Librería | Alias | Función en el taller |
|---|---|---|
| **NumPy** | `np` | Cálculos numéricos sobre arreglos. Se usa para representar valores faltantes (`np.nan`). |
| **pandas** | `pd` | Manipulación de datos en tablas llamadas `DataFrame`: leer archivos, filtrar, agrupar y resumir. |
| **Matplotlib** | `plt` | Creación y personalización de gráficos. |
| **Seaborn** | `sns` | Gráficos estadísticos con menos código y mejor apariencia. Está construida sobre Matplotlib. |
| **scikit-learn** | `sklearn` | Aprendizaje automático: división de datos, preprocesamiento, modelos y métricas. |

De scikit-learn se importan solo los elementos necesarios:

| Elemento | Para qué sirve |
|---|---|
| `train_test_split` | Divide los datos en entrenamiento y prueba. |
| `cross_val_score` | Evalúa un modelo con validación cruzada. |
| `Pipeline`, `ColumnTransformer` | Encadenan el preprocesamiento y el modelo en un solo objeto. |
| `SimpleImputer` | Rellena valores faltantes. |
| `StandardScaler` | Lleva las variables numéricas a una misma escala. |
| `OneHotEncoder` | Convierte variables de texto en columnas numéricas. |
| `DummyClassifier` | Modelo de referencia que no aprende; sirve como punto de comparación. |
| `LogisticRegression` | Modelo lineal de clasificación. |
| `RandomForestClassifier` | Modelo que combina muchos árboles de decisión. |
| Funciones de `sklearn.metrics` | Miden el desempeño del modelo. |

En la celda siguiente, `import` carga una librería, `as` le asigna un alias corto y `from ... import ...` carga solo elementos específicos. `SEMILLA = 42` fija el componente aleatorio para que todos los estudiantes obtengan los mismos resultados.
""")

code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, ConfusionMatrixDisplay,
                             RocCurveDisplay, roc_auc_score, accuracy_score, recall_score)

SEMILLA = 42
sns.set_theme(style="whitegrid")
print("Librerías cargadas correctamente.")
""")

# 2. CARGA
md(r"""
## 2. Carga y comprensión de los datos

El archivo `desercion_estudiantil.csv` está en la carpeta `datos/` del repositorio de GitHub. Un archivo **CSV** es un archivo de texto en el que cada línea es un registro y los valores se separan con comas.

La función `pd.read_csv()` lee el archivo desde la dirección *raw* de GitHub (que entrega el contenido puro del archivo) y lo guarda en la variable `df`. Antes de ejecutar, se debe reemplazar `USUARIO` por el nombre de la cuenta de GitHub donde se publicó el repositorio.

Si la descarga falla, la celda permite subir el archivo manualmente desde el computador con `files.upload()`, una función propia de Colab.
""")

code(r"""
USUARIO_GITHUB = "USUARIO"   # Reemplazar por el usuario de GitHub
URL = f"https://raw.githubusercontent.com/{USUARIO_GITHUB}/taller-analisis-predictivo/main/datos/desercion_estudiantil.csv"

try:
    df = pd.read_csv(URL)
except Exception:
    try:
        df = pd.read_csv("../datos/desercion_estudiantil.csv")   # Ejecución local
    except FileNotFoundError:
        from google.colab import files
        files.upload()                                          # Subir el CSV manualmente
        df = pd.read_csv("desercion_estudiantil.csv")

print("Filas y columnas:", df.shape)
""")

md(r"""
### Diccionario de datos

| Variable | Descripción |
|---|---|
| `id_estudiante` | Código del estudiante. No se usa para predecir. |
| `edad` | Edad en años. |
| `genero` | Femenino o Masculino. |
| `estrato` | Estrato socioeconómico (1 a 6). |
| `modalidad` | Presencial o Virtual. |
| `trabaja` | Indica si trabaja (Si / No). |
| `horas_trabajo_semana` | Horas semanales de trabajo. |
| `beca` | Indica si tiene beca (Si / No). |
| `horas_estudio_semana` | Horas semanales de estudio independiente. |
| `asistencia_pct` | Porcentaje de asistencia a clases. |
| `materias_perdidas` | Número de asignaturas reprobadas. |
| `satisfaccion` | Satisfacción con la institución (1 a 5). |
| `promedio_acumulado` | Promedio académico (0.0 a 5.0). |
| `deserta` | **Variable objetivo:** 1 si desertó, 0 si permaneció. |

**Conceptos clave:** la *variable objetivo* es lo que se quiere predecir; las *variables predictoras* son la información que el modelo usa para predecir. Como la variable objetivo es una categoría (deserta o no), se trata de un problema de **clasificación**.

> Los datos son sintéticos: fueron generados por computador con fines educativos y no corresponden a personas reales.
""")

# 3. EXPLORACIÓN
md(r"""
## 3. Exploración inicial

`df.head()` muestra las primeras cinco filas de la tabla para tener una primera impresión de los datos.
""")

code(r"""
df.head()
""")

md(r"""
`df.info()` muestra el nombre de cada columna, cuántos valores no nulos tiene y su tipo de dato (`int64` para enteros, `float64` para decimales y `object` o `str` para texto). Si una columna tiene menos valores no nulos que el total de filas, tiene datos faltantes.
""")

code(r"""
df.info()
""")

md(r"""
`df.describe()` calcula estadísticos de las columnas numéricas: conteo, media, desviación estándar, mínimo, cuartiles y máximo. Se agrega `.T` para mostrar las variables como filas. Conviene buscar valores sin sentido, como edades de 5 o 150 años.
""")

code(r"""
df.describe().T
""")

md(r"""
La siguiente celda revisa tres aspectos de calidad:

- `df.isna().sum()`: cuenta los valores faltantes por columna.
- `df.duplicated().sum()`: cuenta las filas repetidas.
- `value_counts()`: cuenta cuántas veces aparece cada valor. Se aplica a `trabaja` para revisar cómo están escritas sus categorías y a `deserta` para conocer la proporción de cada clase.
""")

code(r"""
print("Valores faltantes por columna:")
print(df.isna().sum()[df.isna().sum() > 0])
print("\nFilas duplicadas:", df.duplicated().sum())
print("\nCategorías de 'trabaja':")
print(df["trabaja"].value_counts())
print("\nProporción de cada clase en 'deserta':")
print(df["deserta"].value_counts(normalize=True).round(3))
""")

md(r"""
### Actividad del estudiante 1

1. ¿Cuántos registros y variables tiene el conjunto de datos?
2. ¿Qué problemas de calidad se identificaron (faltantes, duplicados, valores imposibles, escritura inconsistente)?
3. ¿Qué porcentaje de estudiantes desertó?
""")

# 4. LIMPIEZA
md(r"""
## 4. Limpieza de los datos

Se corrigen los tres problemas detectados:

1. **Duplicados:** `drop_duplicates()` elimina las filas repetidas.
2. **Categorías inconsistentes:** `.str.capitalize()` deja la primera letra en mayúscula y el resto en minúscula, de modo que `si` y `Si` quedan iguales.
3. **Edades imposibles:** las edades menores de 15 o mayores de 70 se reemplazan por `np.nan` (valor faltante). Se prefiere esto a eliminar la fila, porque así se conserva el resto de su información. `df.loc[condición, columna] = valor` asigna el valor solo en las filas que cumplen la condición; el operador `|` significa "o".

Los valores faltantes **no** se rellenan todavía. Se rellenarán dentro del *pipeline* (sección 6), usando solo los datos de entrenamiento, para evitar la fuga de información.
""")

code(r"""
df = df.drop_duplicates().reset_index(drop=True)
df["trabaja"] = df["trabaja"].str.capitalize()
df.loc[(df["edad"] < 15) | (df["edad"] > 70), "edad"] = np.nan

print("Filas después de la limpieza:", len(df))
print("Categorías de 'trabaja':", list(df["trabaja"].unique()))
print("Rango de edad:", df["edad"].min(), "a", df["edad"].max())
""")

# 5. EDA
md(r"""
## 5. Análisis exploratorio visual

El análisis visual permite identificar qué variables podrían ayudar a predecir la deserción.

### 5.1 Variables numéricas según la deserción

Se dibuja un diagrama de caja (*boxplot*) por variable, separando a quienes desertan de quienes permanecen. La caja abarca del 25 % al 75 % de los datos y la línea interior es la mediana. Si las cajas de ambos grupos están muy separadas, la variable distingue bien a los grupos.

- `plt.subplots(2, 3)` crea una cuadrícula de 2 filas y 3 columnas de gráficos.
- `zip()` recorre al mismo tiempo cada gráfico y cada variable.
- `sns.boxplot(x=..., y=...)` dibuja el diagrama de caja.
""")

code(r"""
variables = ["promedio_acumulado", "materias_perdidas", "asistencia_pct",
             "horas_estudio_semana", "horas_trabajo_semana", "satisfaccion"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, var in zip(axes.flatten(), variables):
    sns.boxplot(data=df, x="deserta", y=var, ax=ax, palette=["#4C72B0", "#C44E52"])
    ax.set_title(var)
    ax.set_xlabel("deserta (0 = permanece, 1 = deserta)")
plt.tight_layout()
plt.show()
""")

md(r"""
### 5.2 Tasa de deserción por variable categórica

Como `deserta` vale 0 o 1, su promedio dentro de un grupo es la proporción de estudiantes que desertan. `groupby(var)["deserta"].mean()` calcula esa tasa para cada categoría.
""")

code(r"""
categoricas = ["modalidad", "trabaja", "beca", "estrato"]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, var in zip(axes, categoricas):
    df.groupby(var)["deserta"].mean().plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title(f"Tasa de deserción por {var}")
    ax.set_ylabel("Proporción que deserta")
    ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()
""")

md(r"""
### 5.3 Correlación con la variable objetivo

La **correlación** mide la relación lineal entre dos variables numéricas, de -1 a 1. Un valor positivo indica que, al aumentar la variable, aumenta la deserción; un valor negativo indica lo contrario. `corr(numeric_only=True)` calcula las correlaciones solo entre columnas numéricas.
""")

code(r"""
correlacion = df.corr(numeric_only=True)["deserta"].drop("deserta").sort_values()
correlacion.plot(kind="barh", color=["#C44E52" if c > 0 else "#4C72B0" for c in correlacion], figsize=(8, 5))
plt.title("Correlación de cada variable con la deserción")
plt.axvline(0, color="black", linewidth=0.8)
plt.show()
""")

md(r"""
### Actividad del estudiante 2

1. ¿Qué tres variables parecen más relacionadas con la deserción? Justificar con los gráficos.
2. ¿Qué grupos (modalidad, trabajo, beca, estrato) presentan mayor deserción? Proponer una posible explicación.
""")

# 6. PREPARACIÓN
md(r"""
## 6. Preparación para el modelado

### 6.1 Separación de variables y división de los datos

Por convención, `X` contiene las variables predictoras y `y` la variable objetivo.

Luego, `train_test_split()` divide los datos en dos partes:
- **Entrenamiento (80 %):** el modelo aprende de estos datos.
- **Prueba (20 %):** se reservan para evaluar el modelo al final con datos que nunca vio.

El parámetro `stratify=y` mantiene la misma proporción de desertores en ambas partes.
""")

code(r"""
numericas = ["edad", "estrato", "horas_trabajo_semana", "horas_estudio_semana",
             "asistencia_pct", "materias_perdidas", "satisfaccion", "promedio_acumulado"]
categoricas = ["genero", "modalidad", "trabaja", "beca"]

X = df[numericas + categoricas]
y = df["deserta"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEMILLA)

print("Entrenamiento:", X_train.shape, "| Prueba:", X_test.shape)
""")

md(r"""
### 6.2 Pipeline de preprocesamiento

Los modelos requieren datos numéricos y sin faltantes. El `ColumnTransformer` aplica un tratamiento distinto a cada tipo de columna:

**Numéricas:**
1. `SimpleImputer(strategy="median")` rellena faltantes con la mediana.
2. `StandardScaler()` transforma cada variable para que tenga media 0 y desviación 1, de modo que todas queden en la misma escala.

**Categóricas:**
1. `SimpleImputer(strategy="most_frequent")` rellena faltantes con la categoría más común.
2. `OneHotEncoder()` convierte cada categoría en una columna de 0 y 1. Por ejemplo, `beca` se convierte en `beca_No` y `beca_Si`.

Al poner este preprocesamiento dentro de un `Pipeline` junto con el modelo, scikit-learn calcula las medianas y escalas **solo con los datos de entrenamiento**. Esto evita la **fuga de información**, es decir, que el modelo "vea" por accidente los datos de prueba.
""")

code(r"""
preprocesador = ColumnTransformer([
    ("num", Pipeline([("imputar", SimpleImputer(strategy="median")),
                      ("escalar", StandardScaler())]), numericas),
    ("cat", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                      ("codificar", OneHotEncoder(handle_unknown="ignore"))]), categoricas)
])
preprocesador
""")

# 7. MODELOS
md(r"""
## 7. Entrenamiento y comparación de modelos

Se comparan tres modelos:

| Modelo | Idea principal |
|---|---|
| **Línea base** (`DummyClassifier`) | Siempre predice la clase más frecuente. No aprende nada; cualquier modelo útil debe superarlo. |
| **Regresión logística** | Combina las variables con pesos y convierte el resultado en una probabilidad. Es simple e interpretable. |
| **Bosque aleatorio** | Entrena muchos árboles de decisión con muestras aleatorias y combina sus votos. Capta relaciones más complejas. |

El parámetro `class_weight="balanced"` da más peso a la clase menos frecuente (los desertores).

### Validación cruzada

En lugar de evaluar con una sola división, `cross_val_score(..., cv=5)` divide el entrenamiento en 5 partes, entrena 5 veces dejando una parte distinta para validar y devuelve los 5 resultados. Su promedio es una estimación más confiable del desempeño.

La métrica usada es **ROC-AUC**: mide qué tan bien el modelo ordena a los estudiantes de mayor a menor riesgo. Un valor de 0.5 equivale al azar y 1.0 es perfecto.
""")

code(r"""
modelos = {
    "Línea base": DummyClassifier(strategy="most_frequent"),
    "Regresión logística": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Bosque aleatorio": RandomForestClassifier(n_estimators=200, max_depth=6,
                                               class_weight="balanced", random_state=SEMILLA)
}

for nombre, modelo in modelos.items():
    pipeline = Pipeline([("prep", preprocesador), ("modelo", modelo)])
    puntajes = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"{nombre:22s} ROC-AUC = {puntajes.mean():.3f} (+/- {puntajes.std():.3f})")
""")

md(r"""
Se elige como modelo final la **regresión logística**, que obtiene un ROC-AUC prácticamente igual al del bosque aleatorio y además es más fácil de interpretar. Entre dos modelos de desempeño similar, conviene el más sencillo.

`fit()` entrena el pipeline completo con todos los datos de entrenamiento.
""")

code(r"""
modelo_final = Pipeline([("prep", preprocesador),
                         ("modelo", LogisticRegression(max_iter=1000, class_weight="balanced"))])
modelo_final.fit(X_train, y_train)
print("Modelo entrenado.")
""")

# 8. EVALUACIÓN
md(r"""
## 8. Evaluación e interpretación

### 8.1 Métricas en los datos de prueba

Ahora se evalúa el modelo con los datos de prueba, que no se usaron en el entrenamiento.

- `predict()` devuelve la clase predicha (0 o 1).
- `predict_proba()[:, 1]` devuelve la probabilidad de desertar.

| Métrica | Pregunta que responde |
|---|---|
| **Exactitud** (*accuracy*) | ¿Qué porcentaje de las predicciones fue correcto? |
| **Precisión** (*precision*) | De los que el modelo señala como desertores, ¿cuántos realmente desertan? |
| **Sensibilidad** (*recall*) | De los que realmente desertan, ¿cuántos detecta el modelo? |
| **F1** | Equilibrio entre precisión y sensibilidad. |

En este problema la **sensibilidad** es especialmente importante: no detectar a un estudiante en riesgo es más costoso que ofrecer apoyo a alguien que no lo necesitaba.
""")

code(r"""
y_pred = modelo_final.predict(X_test)
y_prob = modelo_final.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["Permanece", "Deserta"], digits=3))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
""")

md(r"""
### 8.2 Matriz de confusión y curva ROC

- La **matriz de confusión** muestra cuántos estudiantes de cada clase real fueron clasificados en cada clase predicha. La diagonal contiene los aciertos.
- La **curva ROC** muestra el compromiso entre detectar desertores y generar falsas alarmas. Cuanto más se acerque a la esquina superior izquierda, mejor. La línea punteada representa el azar.
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Permanece", "Deserta"],
                                        cmap="Blues", ax=axes[0], colorbar=False)
axes[0].set_title("Matriz de confusión")
RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[1], name="Regresión logística")
axes[1].plot([0, 1], [0, 1], "k--")
axes[1].set_title("Curva ROC")
plt.tight_layout()
plt.show()
""")

md(r"""
### 8.3 Variables que más influyen

Cada variable tiene un **coeficiente** en la regresión logística:
- **Positivo (rojo):** al aumentar la variable, aumenta el riesgo de deserción.
- **Negativo (azul):** al aumentar la variable, el riesgo disminuye.

Como las variables numéricas están estandarizadas, los coeficientes se pueden comparar entre sí. Se accede a ellos con `named_steps["modelo"].coef_`, y a los nombres de las columnas con `get_feature_names_out()`.
""")

code(r"""
coeficientes = pd.Series(
    modelo_final.named_steps["modelo"].coef_[0],
    index=modelo_final.named_steps["prep"].get_feature_names_out()
).sort_values()
coeficientes.index = coeficientes.index.str.replace("num__", "").str.replace("cat__", "")

coeficientes.plot(kind="barh", figsize=(8, 7),
                  color=["#C44E52" if c > 0 else "#4C72B0" for c in coeficientes])
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Coeficientes de la regresión logística")
plt.show()
""")

md(r"""
### Actividad del estudiante 3

1. ¿Qué modelo obtuvo mejor ROC-AUC en la validación cruzada? ¿Cuánto supera a la línea base?
2. Según la matriz de confusión, ¿cuántos desertores reales no fueron detectados?
3. ¿Qué variables aumentan más el riesgo de deserción y cuáles lo reducen?
""")

# 9. PREDICCIÓN
md(r"""
## 9. Predicción para un estudiante nuevo

El modelo entrenado puede estimar el riesgo de un estudiante que no estaba en los datos. Se crea un `DataFrame` con una fila y las mismas columnas usadas en el entrenamiento, y se aplica `predict_proba()`. Como el pipeline incluye el preprocesamiento, los datos se ingresan en su formato original.
""")

code(r"""
nuevo = pd.DataFrame([{
    "edad": 24, "estrato": 2, "horas_trabajo_semana": 35, "horas_estudio_semana": 4,
    "asistencia_pct": 70, "materias_perdidas": 3, "satisfaccion": 2, "promedio_acumulado": 2.8,
    "genero": "Masculino", "modalidad": "Virtual", "trabaja": "Si", "beca": "No"
}])

probabilidad = modelo_final.predict_proba(nuevo)[0, 1]
print(f"Probabilidad de deserción: {probabilidad:.1%}")
print("Nivel de riesgo:", "Alto" if probabilidad >= 0.6 else "Medio" if probabilidad >= 0.3 else "Bajo")
""")

md(r"""
### Actividad del estudiante 4

1. Modificar los datos del estudiante nuevo (por ejemplo, asignarle beca o aumentar su asistencia al 90 %) y volver a ejecutar la celda. ¿Cómo cambia la probabilidad?
2. ¿Qué acciones podría tomar la universidad con base en estos resultados?
""")

# 10. CIERRE
md(r"""
## 10. Actividades de cierre

### Resumen del proceso

En este taller se recorrió el ciclo de un análisis predictivo: comprensión del problema, exploración, limpieza, análisis visual, preparación con pipeline, comparación de modelos con validación cruzada, evaluación, interpretación y predicción para casos nuevos.

### Reflexión ética

Un modelo de deserción debe usarse para **ofrecer apoyo**, nunca para excluir o estigmatizar. Variables como el estrato pueden reflejar desigualdades, y la predicción debe ser un insumo para que una persona (consejero o docente) tome la decisión final.

### Entregable

El estudiante deberá entregar el cuaderno ejecutado con las cuatro actividades respondidas y un párrafo final de conclusiones (máximo 200 palabras) dirigido a las directivas de la universidad.

### Criterios de evaluación

| Criterio | Peso |
|---|---|
| Ejecución completa y correcta del cuaderno | 25 % |
| Respuestas de exploración y limpieza (actividades 1 y 2) | 25 % |
| Interpretación del modelo y sus métricas (actividad 3) | 25 % |
| Análisis de escenarios y conclusiones (actividad 4 y párrafo final) | 25 % |

### Referencias

- Documentación de pandas: https://pandas.pydata.org/docs/
- Documentación de scikit-learn: https://scikit-learn.org/stable/user_guide.html
- Documentación de Seaborn: https://seaborn.pydata.org/
""")

nb = nbf.v4.new_notebook()
nb["cells"] = celdas
nb["metadata"] = {
    "colab": {"provenance": [], "toc_visible": True},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, "notebooks/taller_analisis_predictivo.ipynb")
print("Cuaderno generado con", len(celdas), "celdas.")
