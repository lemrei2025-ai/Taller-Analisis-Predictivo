"""
Genera el conjunto de datos sintetico del taller: desercion_estudiantil.csv

Simula informacion de 1.200 estudiantes universitarios. La variable objetivo
es 'deserta' (1 = deserto, 0 = permanecio). Se incluyen de forma intencional
valores faltantes, duplicados, edades imposibles y categorias mal escritas
para practicar la limpieza de datos.
"""
import numpy as np
import pandas as pd

SEMILLA = 42
N = 1200
rng = np.random.default_rng(SEMILLA)

edad = np.clip(rng.normal(21, 3, N).round(), 16, 45)
genero = rng.choice(["Femenino", "Masculino"], N, p=[0.52, 0.48])
estrato = rng.choice([1, 2, 3, 4, 5, 6], N, p=[0.18, 0.30, 0.28, 0.14, 0.06, 0.04])
modalidad = rng.choice(["Presencial", "Virtual"], N, p=[0.7, 0.3])
trabaja = rng.choice(["Si", "No"], N, p=[0.42, 0.58])
horas_trabajo = np.where(trabaja == "Si", rng.normal(28, 9, N).clip(5, 60), 0).round()
beca = rng.choice(["Si", "No"], N, p=[0.22, 0.78])
horas_estudio = np.clip(rng.gamma(3, 3.2, N) - 0.08 * horas_trabajo, 0, 40).round(1)
asistencia = np.clip(rng.normal(85, 10, N) - 0.15 * horas_trabajo, 30, 100).round(1)
materias_perdidas = rng.poisson(np.clip(1.6 - 0.04 * horas_estudio - 0.01 * (asistencia - 80), 0.05, None))
satisfaccion = np.clip(rng.normal(3.6, 0.9, N), 1, 5).round()

promedio = np.clip(
    2.6 + 0.035 * horas_estudio + 0.012 * (asistencia - 80) - 0.18 * materias_perdidas
    + 0.15 * (beca == "Si") + 0.04 * satisfaccion + 0.0015 * 40 * (estrato - 2.5)
    + rng.normal(0, 0.3, N), 0.5, 5.0).round(2)

logit = (-1.9 - 1.9 * (promedio - 3.3) + 0.28 * materias_perdidas + 0.025 * horas_trabajo
         - 0.03 * (asistencia - 80) - 0.25 * (estrato - 3) - 0.8 * (beca == "Si")
         - 0.35 * (satisfaccion - 3.5) + 0.35 * (modalidad == "Virtual"))
deserta = rng.binomial(1, 1 / (1 + np.exp(-logit)))

df = pd.DataFrame({
    "id_estudiante": [f"EST{i:04d}" for i in range(1, N + 1)],
    "edad": edad, "genero": genero, "estrato": estrato, "modalidad": modalidad,
    "trabaja": trabaja, "horas_trabajo_semana": horas_trabajo, "beca": beca,
    "horas_estudio_semana": horas_estudio, "asistencia_pct": asistencia,
    "materias_perdidas": materias_perdidas, "satisfaccion": satisfaccion,
    "promedio_acumulado": promedio, "deserta": deserta,
})

# Imperfecciones intencionales
for col, frac in [("horas_estudio_semana", 0.05), ("asistencia_pct", 0.04), ("satisfaccion", 0.03)]:
    idx = rng.choice(N, int(N * frac), replace=False)
    df.loc[idx, col] = np.nan
df.loc[rng.choice(N, 6, replace=False), "edad"] = [99, 120, 5, 99, 8, 150]
df.loc[rng.choice(N, 20, replace=False), "trabaja"] = "si"
df = pd.concat([df, df.sample(10, random_state=SEMILLA)], ignore_index=True)

df.to_csv("datos/desercion_estudiantil.csv", index=False)
print(df.shape, "tasa desercion:", round(df["deserta"].mean(), 3))
