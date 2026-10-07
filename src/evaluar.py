"""Evaluador comun para cualquier modelo (LightGBM, FLAML, AutoGluon, XGBoost...).

Regla anti-ruido (igual para todos):
  1. Seleccion: folds 202607-202610 x semillas 42,43,44, delta PAREADO vs limpio_reg.
  2. Sellado: fold 202611, mismas semillas, delta pareado vs limpio_reg.
  3. Veredicto "MEJORA" solo si delta_sel > 0.0043 (DMD), mejora en >=3/4 folds
     y delta_sellado > 0. Si no, "RUIDO" (o "PEOR" si delta_sel < -0.0043).
  4. Cualquier busqueda de hiperparametros (AutoML) se hace SOLO con meses <= 202606
     (entrena < 202606, valida 202606). Asi los folds de seleccion no se contaminan.

Uso desde un script:
    from src.evaluar import evaluar
    def fit_predict(tr, va, seed):   # DataFrames con features + objetivo
        ...; return probas_de_va
    evaluar("mi_modelo", fit_predict, transformar=None, nota="...")
"""
from __future__ import annotations

import csv
import json
import os
import time
from datetime import date

import numpy as np

from src import config as C
from src import experiment as E
from src.data import load_all
from src.validation import folds, reporte
from src.variantes import _limpio

SEEDS = [42, 43, 44]
REF_SEL = "resultados/limpio_reg.json"
REF_SELLADO = "resultados/sellado_limpio_reg.json"
REGISTRO = "resultados/registro.csv"
DMD = 0.0043


def _veredicto(sel, sellado):
    g, s = sel["global"], sellado["global"]
    ok_folds = int(g["mejora_folds"].split("/")[0]) >= 3
    if g["delta_media"] > DMD and ok_folds and s["delta_media"] > 0:
        return "MEJORA"
    if g["delta_media"] < -DMD:
        return "PEOR"
    return "RUIDO"


def evaluar(nombre, fit_predict, *, transformar=None, filtro_train=None,
            seeds=SEEDS, nota="", guardar_oof=True):
    t0 = time.time()
    df = (transformar or _limpio)(load_all())
    train = df[df["_origen"] == "train"]
    feats = E.columnas_features(df)
    res = {}
    oof = {}
    for m, tr, va in folds(train, E.FOLDS_SEL + [E.FOLD_SELLADO]):
        if filtro_train is not None:
            tr = tr[filtro_train(tr)]
        for s in seeds:
            p = np.asarray(fit_predict(tr[feats + [C.TARGET, C.MONTH]].copy(),
                                       va[feats + [C.MONTH]].copy(), s), dtype=float)
            res.setdefault(s, {})[m] = reporte(va, p)
            oof.setdefault(m, []).append(p)
        print(f"  [{nombre}] {m}: {res[seeds[0]][m]}", flush=True)
    sel = {s: {m: r for m, r in rs.items() if m != E.FOLD_SELLADO} for s, rs in res.items()}
    sll = {s: {E.FOLD_SELLADO: rs[E.FOLD_SELLADO]} for s, rs in res.items()}
    os.makedirs("resultados", exist_ok=True)
    params = {"nota": nota}
    E.guardar(f"resultados/{nombre}.json", experimento=nombre, seeds=seeds,
              meses=E.FOLDS_SEL, params=params, resultados=sel, force=True)
    E.guardar(f"resultados/sellado_{nombre}.json", experimento=nombre, seeds=seeds,
              meses=[E.FOLD_SELLADO], params=params, resultados=sll, force=True)
    if guardar_oof:
        os.makedirs("resultados/oof", exist_ok=True)
        np.savez_compressed(f"resultados/oof/{nombre}.npz",
                            **{str(m): np.mean(v, 0) for m, v in oof.items()})
    c_sel = E.comparar(E.cargar(REF_SEL), sel)
    c_sll = E.comparar(E.cargar(REF_SELLADO), sll)
    r_sel, r_sll = E.resumen(sel)["media"]["global"], E.resumen(sll)["media"]["global"]
    ver = _veredicto(c_sel, c_sll)
    fila = {
        "fecha": date.today().isoformat(), "nombre": nombre,
        "gini_sel": r_sel, "delta_sel": c_sel["global"]["delta_media"],
        "delta_sel_sd": c_sel["global"]["delta_sd"],
        "mejora_folds": c_sel["global"]["mejora_folds"],
        "gini_sellado": r_sll, "delta_sellado": c_sll["global"]["delta_media"],
        "veredicto": ver, "segundos": round(time.time() - t0), "nota": nota,
    }
    nuevo = not os.path.exists(REGISTRO)
    with open(REGISTRO, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fila))
        if nuevo:
            w.writeheader()
        w.writerow(fila)
    print(json.dumps(fila, ensure_ascii=False))
    return fila
