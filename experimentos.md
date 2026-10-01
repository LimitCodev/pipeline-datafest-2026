# Experimentos — Datafest 2026

Un experimento = un cambio + su Gini de validación. Nada de KFold aleatorio:
la validación es temporal (entrena meses < m, valida en m).

- **Folds de selección: 202607, 202608, 202609, 202610.**
- **202611 está SELLADO**: no se usa para elegir nada; solo para confirmar
  finalistas (máx. 1 vez por finalista), contra `baseline_ref_sellado.json`.
- Las filas anteriores a 2026-10-01 usaban noviembre en la validación; quedan como
  registro histórico, previo a la regla del fold sellado.
- `gini()`, `folds()` y `reporte()` de `src/validation.py` son la regla de medición:
  no se modifican sin permiso.

| fecha | autor | cambio | Gini global | Gini nuevos | Gini historial | nota |
|-------|-------|--------|-------------|-------------|----------------|------|
| 2026-10-01 | josue | estructura del repo + pipeline base | — | — | — | pendiente: correr con datos reales |
| 2026-09-30 | mimo | baseline real, 5 folds (jul–nov) | 0.2459 | 0.2636 | 0.2403 | por fold (global): 0.2478, 0.2562, 0.2372, 0.2614, 0.2269; sd: global 0.0140, nuevos 0.0445, historial 0.0131 |
| 2026-09-30 | mimo | diff1 estricto por calendario (función de la P6) | 0.2459 | 0.2636 | 0.2403 | Δ=0 en los 5 folds: 0 filas cambian (0 clientes con huecos, 0 duplicados). No aporta; no repetir |
| 2026-09-30 | mimo | quitar es_nuevo_cliente | 0.2413 | 0.2535 | 0.2355 | pareado Δ −0.0046±0.0088, t=−1.17 < 2.776 (n.s.) → se queda la feature |
| 2026-09-30 | mimo | quitar meses_previos+tiene_historial | 0.2430 | 0.2569 | 0.2367 | pareado Δ −0.0029±0.0043, t=−1.52 (n.s.); sd(Δ) 3× menor que sd(Gini) → compensa medir en pareado |
| 2026-10-01 | mimo | **T0** harness: `src/experiment.py` + `src/variantes.py` + CLI con `--ref/--out` a prueba de pisarse (nada se guarda sin `--out`; `baseline_ref.json`/`sub_baseline.csv` jamás se sobrescriben) | — | — | — | `run_baseline.py` ahora es CLI con folds/seeds por flag; `validation.py` intocable |
| 2026-10-01 | mimo | **T1** `baseline_ref.json` congelado: baseline con **5 semillas** × 4 folds de selección (300 árboles, seeds 42–46) | 0.2476 | 0.2734 | 0.2399 | por fold (global): 0.2496, 0.2499, 0.2331, 0.2577; **ruido de semilla** sd=0.0052 (0.0029/0.0049/0.0067/0.0060); **DMD=0.0043** (nulo analítico 0.0043 y empírico 0.0042 coinciden) → umbral regla 5 = **max(0.0043, 0.003) = 0.0043** |

Reglas:

- Solo agregados: nunca pegues filas de datos en este archivo (regla 3).
- Un Gini por fold de selección (jul/ago/sep/oct), la media y el Δ pareado vs
  `baseline_ref.json` (media ± sd y en cuántos folds mejora).
- Las submissions se anotan con el nombre de archivo en "nota".
- Registrar también los experimentos que NO funcionaron (evita repetirlos).
- Ruido medido el 2026-10-01 (5 semillas, `baseline_ref.json`): sd entre semillas
  del Gini global = **0.0052** por fold. El Δ del Gini entre folds sin parear
  (0.0140) NO sirve para decidir; solo sirve el Δ pareado.
- **Δ mínimo detectable (DMD) = 0.0043**; se acepta solo si Δ pareado medio >
  max(DMD, 0.003) = 0.0043 y mejora en ≥3 de 4 folds.
- Una sola feature que suba >0.03 ⇒ sospecha de fuga, auditar antes de aceptar.
