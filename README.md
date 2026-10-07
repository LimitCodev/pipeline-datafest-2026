# pipeline-datafest-2026

Modelo para la **DataFest 2026**: predecir la probabilidad de que cada cliente de un banco convierta en **diciembre de 2026**, usando su historial de enero a noviembre.

| | Gini |
|---|---|
| Validación (julio a octubre) | **0.2636** |
| Noviembre (mes reservado, nunca usado para decidir) | **0.2479** |
| Línea base inicial | 0.2476 / 0.2256 |

El modelo final es un **LightGBM regularizado** con tres variables de reglas de negocio, entrenado sin enero. El detalle está en [`RESULTADOS.md`](RESULTADOS.md).

---

## Contenido

1. [El problema](#el-problema)
2. [Instalación](#instalación)
3. [Generar la entrega](#generar-la-entrega)
4. [Cómo validamos](#cómo-validamos)
5. [Qué hace el modelo final](#qué-hace-el-modelo-final)
6. [Qué probamos y no mejoró](#qué-probamos-y-no-mejoró)
7. [Probar un modelo nuevo](#probar-un-modelo-nuevo)
8. [Redes neuronales en Google Colab](#redes-neuronales-en-google-colab)
9. [Estructura del repositorio](#estructura-del-repositorio)
10. [Reglas del repositorio](#reglas-del-repositorio)

---

## El problema

- **Datos:** `train.csv` tiene 110 100 filas de enero a noviembre de 2026. Cada fila es un cliente en un mes. Un mismo cliente puede aparecer en varios meses.
- **Objetivo:** `objetivo = 1` si el cliente convirtió por primera vez ese mes.
- **Prueba:** `test.csv` tiene 9 900 filas, todas de diciembre de 2026.
- **Métrica:** `Gini = 2 * AUC - 1`. Solo importa **ordenar** bien a los clientes, no calibrar la probabilidad.
- **Entrega:** un CSV con las columnas `id_cliente,prediccion`, una fila por cada fila de `test.csv` y en el mismo orden.

Los datos del concurso **no están en el repositorio**. Hay que copiarlos en `datafest-datos/`:

```text
datafest-datos/
├── train.csv
├── test.csv
└── sample_submission.csv
```

## Instalación

Necesitas Python 3.12 o superior.

```bash
git clone https://github.com/LimitCodev/pipeline-datafest-2026.git
cd pipeline-datafest-2026
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Generar la entrega

1. Valida el modelo final y compáralo con la línea base:

   ```bash
   .venv/bin/python -m src.run_baseline --variante reglas_sin_ene_reg --ref baseline_ref.json
   ```

2. Entrena con enero a noviembre, promedia 5 semillas y escribe la entrega:

   ```bash
   .venv/bin/python -m src.make_submission --variante reglas_sin_ene_reg \
       --seeds 42 43 44 45 46 --out submissions/entrega_final.csv
   ```

3. Comprueba que el archivo cumple el formato de la competencia:

   ```bash
   .venv/bin/python -m src.validar_entrega submissions/entrega_final.csv
   ```

   Revisa columnas exactas, 9 900 filas en el orden de `test.csv`, sin duplicados ni vacíos, probabilidades entre 0 y 1, y saltos de línea CRLF como en `sample_submission.csv`.

`submissions/` no se sube al repositorio porque contiene predicciones por cliente.

## Cómo validamos

Un mismo cliente aparece en varios meses. Si se valida con un corte aleatorio, el modelo ve al cliente en entrenamiento y lo "reconoce" en validación, y el Gini sale inflado. Por eso validamos **por tiempo**:

- **Para validar un mes, entrenamos solo con los meses anteriores.** Así simulamos predecir diciembre con enero a noviembre.
- **Meses de selección:** 202607, 202608, 202609 y 202610, cada uno con 3 semillas. Todas las decisiones se toman con estos meses.
- **Mes reservado:** 202611. No se usa para elegir nada; solo confirma el modelo final.

Un cambio se acepta solo si cumple **las tres** condiciones:

1. Supera al modelo de referencia en más de **0.0043** de Gini, comparando cada semilla con la misma semilla. Ese umbral es el ruido medido entre semillas.
2. Mejora en al menos 3 de los 4 meses.
3. También mejora en noviembre.

Todo modelo pasa por el mismo juez, `src/evaluar.py`. Cada prueba queda como una fila en `resultados/registro.csv`, con su veredicto: `MEJORA`, `RUIDO` o `PEOR`.

## Qué hace el modelo final

La variante se llama `reglas_sin_ene_reg` y está definida en `src/variantes.py`.

1. **LightGBM regularizado** (`PARAMS_REG`): árboles de 7 hojas, al menos 300 clientes por hoja, tasa de aprendizaje 0.01 y regularización L2. Aporta +0.012 de Gini, la mayor parte de la mejora.
2. **Quita columnas de cambio mes a mes que siempre valen 0.** Todas las variables son fijas por cliente, salvo `dias_ultima_interaccion`.
3. **Añade tres indicadores de reglas** encontradas en el análisis exploratorio:
   - riesgo bajo con 3 o más productos;
   - riesgo alto sin transacciones en más de 180 días;
   - tarjeta de crédito con la app móvil activa.
4. **Entrena sin enero.** En enero ningún cliente tiene meses previos, porque ahí empiezan los datos.
5. **Promedia 5 semillas** en la entrega.

> **Advertencia:** las reglas del punto 3 salieron de un análisis exploratorio que también miró noviembre, así que el 0.2479 de noviembre puede estar algo inflado. La versión sin reglas, `limpio_reg`, no tiene ese sesgo: da 0.2607 en validación y 0.2361 en noviembre.

## Qué probamos y no mejoró

| Prueba | Gini validación | Veredicto |
|---|---|---|
| CatBoost | 0.2510 | peor |
| Regresión logística con splines | 0.2357 | peor |
| FLAML AutoML (LightGBM, XGBoost, ExtraTrees) | 0.2601 / 0.2558 / 0.2539 | ruido o peor |
| AutoGluon con validación aleatoria | 0.2269 | peor |
| AutoGluon con validación por mes | 0.2581 | ruido |
| Historial de interacciones | 0.2587 | peor |
| Codificación de categorías, monotonía, ratios, bagging | 0.2601–0.2616 | ruido |
| `id_cliente` como variable | 0.2608 / 0.2583 | ruido |
| RealMLP (red neuronal tabular) | 0.2237 | peor; en la mezcla con LightGBM su peso óptimo es 0 |
| TabICL y TabM | pendiente | correr en GPU, ver [Colab](#redes-neuronales-en-google-colab) |

Cinco familias de modelos y dos herramientas de AutoML se quedan alrededor de 0.26. Parece el techo de la información que traen los datos. El detalle de cada prueba está en [`experimentos.md`](experimentos.md).

## Probar un modelo nuevo

Escribe una función que reciba el entrenamiento, la validación y una semilla, y que devuelva probabilidades. El juez común hace el resto:

```python
from src.evaluar import evaluar

def mi_modelo(tr, va, seed):
    # tr: features + objetivo + mes; va: features + mes
    ...
    return probabilidades_de_va

evaluar("mi_modelo", mi_modelo, nota="qué cambia")
```

El juez entrena en los 4 meses de selección y en noviembre, compara contra `limpio_reg` y escribe el veredicto en `resultados/registro.csv`. Si buscas hiperparámetros, usa **solo meses hasta 202606**, para no contaminar los meses de selección.

## Redes neuronales en Google Colab

TabICL y TabM son redes neuronales pensadas para GPU. En una laptop con CPU de 16 hilos, TabICL no terminó ni un mes en 20 minutos. En Colab con GPU es mucho más rápido. `src/exp_neuronales.py` usa la GPU automáticamente si existe.

1. En Colab, abre **Entorno de ejecución → Cambiar tipo de entorno → GPU T4**.
2. Clona el repositorio e instala las librerías:

   ```bash
   !git clone https://github.com/LimitCodev/pipeline-datafest-2026.git
   %cd pipeline-datafest-2026
   !pip install -q pytabkit tabicl lightgbm
   ```

3. Crea la carpeta `datafest-datos/` y sube `train.csv` y `test.csv` con el panel de archivos de Colab.
4. Corre los modelos **uno por uno**, nunca a la vez:

   ```bash
   !python -m src.exp_neuronales tabicl
   !python -m src.exp_neuronales tabm
   ```

5. Descarga `resultados/nn_tabicl.json`, `resultados/nn_tabm.json` y `resultados/registro.csv`, y compáralos con el modelo final (0.2636).

## Estructura del repositorio

```text
├── README.md               este archivo
├── RESULTADOS.md           resumen de resultados para el grupo
├── experimentos.md         bitácora de cada experimento, incluidos los fallidos
├── baseline_ref.json       línea base congelada (5 semillas), no se sobrescribe
├── datafest-datos/         datos del concurso (no versionados)
├── resultados/             métricas agregadas por prueba, sin datos de clientes
│   ├── registro.csv        una fila por prueba con Gini y veredicto
│   ├── flaml/              configuraciones encontradas por FLAML
│   └── autogluon/          tablas de modelos de AutoGluon por mes
├── submissions/            entregas generadas (no versionadas)
└── src/
    ├── config.py           rutas y nombres de columnas
    ├── data.py             carga de train y test
    ├── features.py         variables derivadas
    ├── variantes.py        variantes del modelo, incluida la final
    ├── validation.py       folds temporales y reporte de Gini
    ├── experiment.py       guardar, cargar y comparar resultados
    ├── evaluar.py          juez común para cualquier modelo
    ├── run_baseline.py     valida una variante de LightGBM
    ├── make_submission.py  genera la entrega
    ├── validar_entrega.py  comprueba el formato de la entrega
    ├── automl_flaml.py     pruebas con FLAML
    ├── automl_autogluon.py pruebas con AutoGluon
    ├── exp_encuadre.py     pruebas de variables y de encuadre
    ├── exp_neuronales.py   RealMLP, TabICL y TabM
    └── blend.py            mezcla de modelos por rangos
```

## Reglas del repositorio

- **Nunca se suben datos de clientes:** ni `train.csv`, ni `test.csv`, ni predicciones por cliente. En `resultados/` solo hay métricas agregadas.
- **`baseline_ref.json` no se sobrescribe.** Es la referencia contra la que se mide todo.
- **Se registran también los experimentos que fallan,** para no repetirlos.
- **Una sola variable que suba el Gini más de 0.03 es sospechosa de fuga de información** y se audita antes de aceptarla.
