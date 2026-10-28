# Guía del docente

## Taller: Análisis predictivo a partir de un conjunto de datos

### Datos generales

| Aspecto | Descripción |
|---|---|
| Duración | 2 horas |
| Modalidad | Individual o en parejas |
| Herramienta | Google Colab |
| Tiempo de ejecución del cuaderno | Menos de 1 minuto |

### Preparación previa

1. Publicar este repositorio en GitHub de forma **pública**.
2. Reemplazar `USUARIO` en la variable `USUARIO_GITHUB` del cuaderno (sección 2) y en el enlace del `README.md`.
3. Ejecutar el cuaderno una vez en Colab para verificarlo.
4. Compartir el enlace de Colab y la guía del estudiante.

### Planeación sugerida

| Tiempo | Actividad | Secciones |
|---|---|---|
| 15 min | Presentación del caso, Colab y librerías | 1, 2 |
| 25 min | Exploración y limpieza; Actividad 1 | 3, 4 |
| 20 min | Análisis visual; Actividad 2 | 5 |
| 20 min | Preparación y modelado | 6, 7 |
| 25 min | Evaluación e interpretación; Actividad 3 | 8 |
| 15 min | Predicción y cierre; Actividad 4 | 9, 10 |

### Resultados esperados

- 1.210 registros, 10 duplicados, 1.200 tras la limpieza; 6 edades imposibles y 21 registros con `si` en minúscula.
- Faltantes en `horas_estudio_semana` (60), `asistencia_pct` (48) y `satisfaccion` (36).
- Tasa de deserción cercana al 40 %.
- ROC-AUC en validación cruzada: línea base 0.50, regresión logística 0.78, bosque aleatorio 0.78.
- En prueba: ROC-AUC 0.81 y sensibilidad 0.76 para la clase "deserta".
- Variables de mayor riesgo: materias perdidas, horas de trabajo y modalidad virtual. Variables protectoras: promedio acumulado, beca, satisfacción y estrato.
- El estudiante nuevo del ejemplo obtiene una probabilidad cercana al 92 % (riesgo alto).

### Orientaciones

- **Actividad 3:** destacar que la regresión logística y el bosque aleatorio obtienen un desempeño similar, por lo que se prefiere el modelo más simple e interpretable.
- **Actividad 4:** aclarar que el modelo identifica asociaciones, no causas: un cambio simulado no garantiza el mismo efecto en la realidad.

### Errores frecuentes

| Error | Solución |
|---|---|
| `NameError` | Ejecutar las celdas anteriores (**Entorno de ejecución > Ejecutar anteriores**). |
| No carga el CSV | Verificar `USUARIO_GITHUB` y que el repositorio sea público; o subir el archivo cuando la celda lo solicite. |
