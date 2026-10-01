# Experimentos — Datafest 2026

Un experimento = un cambio + su Gini de validación. Nada de KFold aleatorio:
la validación es temporal (entrena meses < m, valida mes m: sep, oct, nov).

| fecha | autor | cambio | Gini global | Gini nuevos | Gini historial | nota |
|-------|-------|--------|-------------|-------------|----------------|------|
| 2026-10-01 | josue | estructura del repo + pipeline base | — | — | — | pendiente: correr con datos reales |

Reglas:

- Solo agregados: nunca pegues filas de datos en este archivo (regla 3).
- Un Gini por fold (sep/oct/nov) y el promedio de los tres.
- Las submissions se anotan con el nombre de archivo en "nota".
- Registrar también los experimentos que NO funcionaron (evita repetirlos).
