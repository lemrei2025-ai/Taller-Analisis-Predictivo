# Guía del estudiante

## Taller: Análisis predictivo a partir de un conjunto de datos

### Propósito

El estudiante construirá un modelo que estime el riesgo de deserción de estudiantes universitarios, recorriendo las etapas de exploración, limpieza, modelado, evaluación y predicción.

### Requisitos

- Cuenta de Google para usar Google Colab.
- Conocimientos básicos de Python y de estadística descriptiva.

### Instrucciones

1. Abrir el enlace del cuaderno compartido por el docente.
2. Guardar una copia: **Archivo > Guardar una copia en Drive**, con el nombre `Taller_Predictivo_Apellido_Nombre`.
3. En la sección 2, reemplazar `USUARIO` por el usuario de GitHub indicado por el docente.
4. Leer cada celda de texto antes de ejecutar la celda de código que la sigue.
5. Ejecutar las celdas en orden con **Shift + Enter**.
6. Responder las actividades en celdas de texto nuevas (**Insertar > Celda de texto**).

### Actividades

| Actividad | Sección | Tema |
|---|---|---|
| 1 | 3 | Diagnóstico de la calidad de los datos |
| 2 | 5 | Variables y grupos asociados a la deserción |
| 3 | 8 | Interpretación del modelo y sus métricas |
| 4 | 9 | Análisis de escenarios y recomendaciones |

### Entregable

Cuaderno ejecutado con las cuatro actividades respondidas y un párrafo final de conclusiones (máximo 200 palabras) dirigido a las directivas de la universidad.

### Evaluación

| Criterio | Peso |
|---|---|
| Ejecución completa y correcta del cuaderno | 25 % |
| Actividades 1 y 2 | 25 % |
| Actividad 3 | 25 % |
| Actividad 4 y conclusiones | 25 % |

### Glosario

| Término | Definición |
|---|---|
| Variable objetivo | Lo que el modelo intenta predecir. |
| Entrenamiento / prueba | Datos para aprender / datos reservados para evaluar. |
| Validación cruzada | Evaluación repetida con distintas particiones de los datos. |
| Pipeline | Secuencia encadenada de preprocesamiento y modelo. |
| Fuga de información | Uso indebido de los datos de prueba durante el entrenamiento. |
| Sensibilidad | Proporción de desertores reales que el modelo detecta. |
| ROC-AUC | Capacidad del modelo para ordenar los casos según su riesgo (0.5 = azar, 1 = perfecto). |
