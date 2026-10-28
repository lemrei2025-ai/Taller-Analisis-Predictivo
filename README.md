# Taller guiado: Análisis predictivo con Python y Google Colab

Repositorio del taller en el que los estudiantes llevarán a cabo un análisis predictivo a partir de un conjunto de datos proporcionado. El caso de estudio aborda la **deserción estudiantil universitaria** y se desarrolla en **Google Colab** en aproximadamente **2 horas**.

## Abrir el taller en Google Colab

Reemplazar `USUARIO` por el nombre de la cuenta de GitHub donde se publicó el repositorio:

```
https://colab.research.google.com/github/USUARIO/taller-analisis-predictivo/blob/main/notebooks/taller_analisis_predictivo.ipynb
```

Antes de ejecutar, el estudiante debe guardar una copia propia (**Archivo > Guardar una copia en Drive**) y, en la sección 2 del cuaderno, reemplazar `USUARIO` en la variable `USUARIO_GITHUB`.

## Contenido del repositorio

```
taller-analisis-predictivo/
├── README.md
├── LICENSE
├── requirements.txt
├── datos/
│   └── desercion_estudiantil.csv          Datos del taller (1.210 registros, 14 variables)
├── notebooks/
│   └── taller_analisis_predictivo.ipynb   Cuaderno guiado para Google Colab
├── docs/
│   ├── guia_estudiante.md                 Instrucciones, actividades y evaluación
│   └── guia_docente.md                    Planeación y resultados esperados
└── scripts/
    ├── generar_datos.py                   Genera el conjunto de datos sintético
    └── construir_notebook.py              Regenera el cuaderno
```

## Estructura del cuaderno

| Sección | Contenido |
|---|---|
| 1 | Preparación del entorno y explicación de las librerías |
| 2 | Carga y comprensión de los datos |
| 3 | Exploración inicial |
| 4 | Limpieza de los datos |
| 5 | Análisis exploratorio visual |
| 6 | Preparación para el modelado (división y pipeline) |
| 7 | Entrenamiento y comparación de modelos con validación cruzada |
| 8 | Evaluación e interpretación |
| 9 | Predicción para un estudiante nuevo |
| 10 | Actividades de cierre y evaluación |

## Datos

Información sintética de 1.200 estudiantes (generada por computador con fines educativos). La variable objetivo es `deserta` (1 = desertó, 0 = permaneció). Incluye intencionalmente valores faltantes, duplicados, edades imposibles y categorías mal escritas para practicar la limpieza.

| Variable | Descripción |
|---|---|
| `id_estudiante` | Código del estudiante |
| `edad` | Edad en años |
| `genero` | Femenino o Masculino |
| `estrato` | Estrato socioeconómico (1 a 6) |
| `modalidad` | Presencial o Virtual |
| `trabaja` | Si / No |
| `horas_trabajo_semana` | Horas semanales de trabajo |
| `beca` | Si / No |
| `horas_estudio_semana` | Horas semanales de estudio |
| `asistencia_pct` | Porcentaje de asistencia |
| `materias_perdidas` | Asignaturas reprobadas |
| `satisfaccion` | Satisfacción con la institución (1 a 5) |
| `promedio_acumulado` | Promedio académico (0.0 a 5.0) |
| `deserta` | Variable objetivo |

## Librerías

NumPy, pandas, Matplotlib, Seaborn y scikit-learn. Todas vienen instaladas en Google Colab.

## Ejecución local (opcional)

```bash
git clone https://github.com/USUARIO/taller-analisis-predictivo.git
cd taller-analisis-predictivo
pip install -r requirements.txt
jupyter notebook notebooks/taller_analisis_predictivo.ipynb
```

## Licencia

Material educativo bajo licencia MIT.
