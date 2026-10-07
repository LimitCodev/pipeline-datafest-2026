# Nuestro resultado Gini

**Gini de validación: 0.2636** (media de jul–oct 2026).
**Gini en noviembre, mes reservado: 0.2479.**
Archivo de entrega: `submissions/entrega_final.csv`. No está en el repositorio porque contiene predicciones por cliente.

| Versión | Gini validación (jul–oct) | Gini noviembre (reservado) |
|---|---|---|
| Línea base inicial | 0.2476 | 0.2256 |
| LightGBM regularizado (`limpio_reg`) | 0.2607 | 0.2361 |
| **Final (`reglas_sin_ene_reg`)** | **0.2636** | **0.2479** |

Mejora sobre la línea base: **+0.016 en validación y +0.022 en noviembre.**

## Cómo medimos

- **Métrica:** `Gini = 2 * AUC - 1`, la misma que usa la competencia.
- **Validación temporal:** para validar un mes, entrenamos solo con los meses anteriores y medimos en ese mes. Así simulamos predecir diciembre usando enero a noviembre.
- **Meses de selección:** 202607, 202608, 202609 y 202610, cada uno con 3 semillas. Las decisiones se toman con estos meses.
- **Mes reservado:** 202611 no se usa para elegir nada. Solo sirve para confirmar el modelo final.
- **Cuándo aceptamos una mejora:** solo si cumple las tres condiciones:
  1. Supera en más de **0.0043** al modelo de referencia, comparando misma semilla con misma semilla. Ese es el ruido medido entre semillas.
  2. Mejora en al menos 3 de los 4 meses.
  3. También mejora en noviembre.
- **Juez común:** `src/evaluar.py` mide todas las pruebas de la misma forma.

## Qué hace el modelo final

1. **LightGBM regularizado** (`PARAMS_REG` en `src/variantes.py`). Usamos árboles de 7 hojas, al menos 300 clientes por hoja, aprendizaje lento y regularización L2. Esto aporta +0.012, la mayor parte de la mejora.
2. **Quitamos columnas de cambio mes a mes que siempre valían 0** (`_limpio`). Todas las variables son fijas por cliente, salvo `dias_ultima_interaccion`.
3. **Agregamos indicadores de reglas encontradas en el análisis exploratorio** (`_reglas`):
   - riesgo bajo con 3 o más productos;
   - riesgo alto sin transacciones en más de 180 días;
   - tarjeta de crédito con la app móvil activa.
4. **Entrenamos sin enero** (`_filtro_sin_ene`). En enero ningún cliente tiene meses previos, porque ahí empiezan los datos.
5. **Generamos la entrega** entrenando con enero a noviembre y promediando 5 semillas.

## Qué probamos y no mejoró

| Prueba | Gini validación | Veredicto |
|---|---|---|
| CatBoost | 0.2510 | peor |
| Regresión logística con splines | 0.2357 | peor |
| Mezcla de modelos por rangos | ≤ 0.2577 | peor |
| FLAML AutoML (LightGBM, XGBoost, ExtraTrees) | 0.2601, 0.2558, 0.2539 | ruido o peor |
| AutoGluon con validación aleatoria | 0.2269 | peor (mezcla al mismo cliente entre entrenamiento y validación) |
| AutoGluon con validación por mes | 0.2581 | ruido |
| Historial de interacciones | 0.2587 | peor |
| Peso por cliente o solo su última fila | 0.2293, 0.1594 | peor |
| Codificación de categorías con meses previos, monotonía, ratios, bagging | 0.2601–0.2616 | ruido |
| `id_cliente` como variable | 0.2608, 0.2583 | ruido |
| RealMLP (red neuronal tabular, `pytabkit`) | 0.2237 | peor; al mezclarlo con LightGBM su peso óptimo es 0 |
| TabM (red neuronal tabular) | — | descartado: no terminó en 15 min por fold en CPU |
| TabICL (modelo fundacional tabular) | — | descartado: necesita más de 8 GB de RAM en CPU |

**Conclusión:** cinco familias de modelos y dos herramientas de AutoML se quedan alrededor de 0.26. Ese parece ser el techo de la información que traen los datos.

En los datos, clientes con el mismo perfil se reparten entre los que convierten y los que no. Por ejemplo, en el grupo más propenso solo convierte el 38 %. El detalle de cada prueba está en `experimentos.md`. Las métricas de cada librería están en `resultados/`:

- `registro.csv`: una fila por prueba, con su Gini y su veredicto.
- `flaml/`: configuraciones encontradas por FLAML.
- `autogluon/`: tabla de modelos de AutoGluon por mes.

En `resultados/` solo hay métricas agregadas, ningún dato de clientes.

## Formato de la entrega (verificado)

La entrega sigue `DATASET_DESCRIPTION.md` y `sample_submission.csv`:

- Columnas exactas: `id_cliente,prediccion`. No incluye `mes`.
- 9 900 filas, en el mismo orden que `test.csv`.
- Cada `id_cliente` aparece una sola vez y no hay valores vacíos.
- `prediccion` es una probabilidad entre 0.025 y 0.445, con punto decimal.
- El archivo es ASCII, sin BOM, con cabecera y con saltos de línea CRLF, igual que `sample_submission.csv`.

Para comprobarlo:

```bash
python -m src.validar_entrega submissions/entrega_final.csv
```

## Cómo reproducirlo

Los datos del concurso no están en el repositorio. Copiar `train.csv`, `test.csv` y `sample_submission.csv` en `datafest-datos/` y ejecutar:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.run_baseline --variante reglas_sin_ene_reg --ref baseline_ref.json
.venv/bin/python -m src.make_submission --variante reglas_sin_ene_reg \
    --seeds 42 43 44 45 46 --out submissions/entrega_final.csv
.venv/bin/python -m src.validar_entrega submissions/entrega_final.csv
```

## Advertencia honesta

Las reglas del punto 3 salieron de un análisis exploratorio que también miró noviembre. Por eso el 0.2479 de noviembre puede estar algo inflado.

La versión sin esas reglas, `limpio_reg`, no tiene ese sesgo. Da 0.2607 en validación y 0.2361 en noviembre.
