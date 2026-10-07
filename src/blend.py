"""Mezcla por rangos de dos modelos usando sus predicciones guardadas en resultados/oof/.

El peso w se elige SOLO con los folds de seleccion (jul-oct); noviembre (sellado)
solo confirma. Uso:
    python -m src.blend lgb_reglas_sin_ene_reg nn_realmlp
"""
import sys

import numpy as np
from scipy.stats import rankdata

from src import config as C
from src import experiment as E
from src.data import load_all
from src.validation import gini


def main(a, b):
    tr = load_all()
    tr = tr[tr["_origen"] == "train"]
    pa, pb = np.load(f"resultados/oof/{a}.npz"), np.load(f"resultados/oof/{b}.npz")
    y = {m: tr.loc[tr[C.MONTH] == m, C.TARGET].values for m in E.FOLDS_SEL + [E.FOLD_SELLADO]}
    r = lambda x: rankdata(x) / len(x)
    mezcla = lambda m, w: (1 - w) * r(pa[str(m)]) + w * r(pb[str(m)])
    sel = lambda w: np.mean([gini(y[m], mezcla(m, w)) for m in E.FOLDS_SEL])
    ws = np.round(np.arange(0, 1.01, 0.05), 2)
    w = max(ws, key=sel)
    for nombre, ww in [(a, 0.0), (b, 1.0), (f"mezcla w={w}", w)]:
        por_fold = [round(gini(y[m], mezcla(m, ww)), 4) for m in E.FOLDS_SEL]
        print(f"{nombre:28s} sel={sel(ww):.4f} {por_fold} sellado={gini(y[E.FOLD_SELLADO], mezcla(E.FOLD_SELLADO, ww)):.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
