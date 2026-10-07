"""Busqueda AutoML con FLAML, sin contaminar los folds de seleccion.

  python -m src.automl_flaml buscar   -> FLAML (<=14 min) entrena mes<202606, valida 202606
  python -m src.automl_flaml evaluar  -> top-3 estimadores por src.evaluar (unico juez)

Salidas en resultados/flaml/: flaml.log, best_configs.json.
"""
import json
import os
import sys

from flaml import AutoML
from flaml.automl.model import (CatBoostEstimator, ExtraTreesEstimator, LGBMEstimator,
                                RandomForestEstimator,
                                XGBoostSklearnEstimator)

from src import config as C
from src import experiment as E
from src.data import load_all
from src.evaluar import evaluar
from src.variantes import _limpio

OUT = "resultados/flaml"
CONFIGS = f"{OUT}/best_configs.json"
MES_VAL = 202606
BUDGET = 840  # s, regla: <= 15 min
CLASES = {"lgbm": LGBMEstimator, "xgboost": XGBoostSklearnEstimator,
          "catboost": CatBoostEstimator, "rf": RandomForestEstimator,
          "extra_tree": ExtraTreesEstimator}  # lrl1 fuera: no acepta NaN fuera de AutoML


def buscar():
    df = _limpio(load_all())
    df = df[(df["_origen"] == "train") & (df[C.MONTH] <= MES_VAL)]
    assert df[C.MONTH].max() == MES_VAL  # nada de >= 202607
    f = E.columnas_features(df)  # mismo X que usa limpio_reg (sin mes)
    tr, va = df[df[C.MONTH] < MES_VAL], df[df[C.MONTH] == MES_VAL]
    os.makedirs(OUT, exist_ok=True)
    a = AutoML()
    a.fit(tr[f], tr[C.TARGET], X_val=va[f], y_val=va[C.TARGET], task="classification",
          metric="roc_auc", time_budget=BUDGET, n_jobs=4, seed=42,
          estimator_list=list(CLASES), log_file_name=f"{OUT}/flaml.log", verbose=1)
    res = {e: {"gini_val_202606": 1 - 2 * a.best_loss_per_estimator[e],
               "config": a.best_config_per_estimator[e]}
           for e in CLASES if a.best_config_per_estimator.get(e)}
    with open(CONFIGS, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    for e, r in sorted(res.items(), key=lambda kv: -kv[1]["gini_val_202606"]):
        print(f"{e}: gini 202606 = {r['gini_val_202606']:.4f}  {r['config']}")


def fp_de(est, config):
    semilla = "random_seed" if est == "catboost" else "random_state"
    def fp(tr, va, seed):
        f = [c for c in tr.columns if c not in (C.TARGET, C.MONTH)]
        kw = {"n_jobs": 4, **config, semilla: seed}
        kw.pop("FLAML_sample_size", None)  # FLAML submuestrea en la busqueda; aqui se entrena con todo
        m = CLASES[est](task="classification", **kw)
        m.fit(tr[f], tr[C.TARGET])
        return m.predict_proba(va[f])[:, 1]
    return fp


def evaluar_top(k=3):
    res = json.load(open(CONFIGS))
    top = sorted(res, key=lambda e: -res[e]["gini_val_202606"])[:k]
    for e in top:
        evaluar(f"flaml_{e}", fp_de(e, res[e]["config"]),
                nota=f"FLAML {e}, busqueda en <=202606: {res[e]['config']}")


if __name__ == "__main__":
    {"buscar": buscar, "evaluar": evaluar_top}[sys.argv[1]]()
