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
| 2026-10-07 | claude | arreglos base: `DATA_DIR` relativo al repo, `run_baseline` sin `--out` ya no falla, `n_jobs=4` (antes 15), `make_submission` respeta `filtro_train` y promedia `--seeds` | — | — | — | baseline reproduce 0.2506 (seed 42) idéntico |
| 2026-10-07 | claude | EDA: todas las columnas son estáticas por cliente salvo `dias_ultima_interaccion`; todo cliente sin fila en dic convirtió | — | — | — | los `diff1` de columnas estáticas son siempre 0/NaN |
| 2026-10-07 | claude | `reg`: LGBM regularizado (lr 0.01, 1000 árb., 7 hojas, min_child 300, colsample 0.5, lambda 10) | 0.2605 | 0.2770 | 0.2553 | Δ +0.0116±0.0047 vs baseline_ref, 4/4 → **aceptado** |
| 2026-10-07 | claude | `limpio_reg`: reg + quitar diff1 estáticos | 0.2607 | 0.2774 | 0.2555 | Δ +0.0118 vs baseline_ref, 4/4. Sellado 202611: 0.2361 (Δ +0.0105 vs 0.2256). `sub_limpio_reg.csv` |
| 2026-10-07 | claude | `inter_reg`: historial de interacciones (lag, mín., media, nº contactos, gap tx−int) | 0.2587 | 0.2749 | 0.2539 | −0.002 vs limpio_reg → **rechazado**, no repetir |
| 2026-10-07 | claude | CatBoost (depth 5, l2 10, 1500 it.) / logística con splines / mezclas por rango | 0.2510 / 0.2357 / ≤0.2577 | — | — | todas peores que LGBM solo → **rechazado** |
| 2026-10-07 | claude | barrido LGBM alrededor de reg (hojas 5/10, min_child 600/1000, lr 0.005, n 700/1500, subsample 0.6, extra_trees, path_smooth, cat_smooth) | 0.2593–0.2612 | — | — | todos \|Δ\| ≤ 0.0014 vs limpio_reg: meseta. colsample 0.3 −0.009 (hay interacciones) |
| 2026-10-07 | claude | `reglas_reg`: flags low&prod≥3, high&tx>180/100, tarjeta&móvil, low&tm&prod≥2 | 0.2620 | 0.2821 | 0.2567 | Δ +0.0012 vs limpio_reg, 2/4 (n.s.) |
| 2026-10-07 | claude | `sin_ruido_reg` (quitar 8 features planas) / pesos recientes hl6, hl12 / sin enero | 0.2609 / 0.2615 / 0.2612 / 0.2620 | — | — | todos n.s. (≤ +0.0012) |
| 2026-10-07 | claude | `reglas_sin_ene_reg`: reglas + sin enero en train | 0.2635 | — | — | Δ +0.0159 vs baseline_ref; +0.0028 vs limpio_reg, 3/4 (bajo DMD). Sellado 202611: 0.2476 (Δ +0.0220). Ojo: las reglas salieron de un EDA que vio noviembre. `sub_reglas_sin_ene_reg.csv` |
| 2026-10-07 | claude | **evaluador común** `src/evaluar.py`: 4 folds × 3 semillas + sellado 202611, Δ pareado vs `limpio_reg`; MEJORA solo si Δ>0.0043, ≥3/4 folds y sellado>0. Registro en `resultados/registro.csv` | — | — | — | control `lgb_limpio_reg` da Δ=0 exacto |
| 2026-10-07 | claude | AutoML FLAML (búsqueda solo con meses ≤202606): lgbm / xgboost / extra_tree | 0.2601 / 0.2558 / 0.2539 | — | — | RUIDO / PEOR / PEOR. `src/automl_flaml.py` |
| 2026-10-07 | claude | AutoGluon 1.6.3 `medium_quality` (validación interna aleatoria) | 0.2269 | — | — | PEOR −0.033: el split aleatorio mezcla al mismo cliente y premia RandomForest |
| 2026-10-07 | claude | AutoGluon `medium_quality` con tuning_data = último mes de train | 0.2581 | — | — | RUIDO (−0.0023, sellado −0.0106). `src/automl_autogluon.py` |
| 2026-10-07 | claude | encuadre: peso 1/filas por cliente / solo última fila / ranking `rank_xendcg` por mes | 0.2293 / 0.1594 / 0.2469 | — | — | PEOR: los supervivientes SÍ son la población evaluada, no repesar. No repetir |
| 2026-10-07 | claude | features: target encoding temporal / monotonía / ordinal+ratios / mes de entrada / bagging 5 LGBM | 0.2601 / 0.2611 / 0.2611 / 0.2616 / 0.2611 | — | — | todos RUIDO (\|Δ\| ≤ 0.0009). `src/exp_encuadre.py` |
| 2026-10-07 | claude | `id_cliente` como feature (numérico / categoría) | 0.2608 / 0.2583 | — | — | RUIDO / RUIDO (sellado −0.003 / −0.008): el ID no lleva información de conversión |

Nota: `baseline_ref_sellado.json` = baseline, 5 semillas, fold 202611 (global 0.2256).


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
