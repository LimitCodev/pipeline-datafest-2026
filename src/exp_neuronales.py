"""Modelos tabulares neuronales recientes (pytabkit) con el juez comun src.evaluar.

Corre en .venv-ag (Python 3.12 + torch CPU):
    .venv-ag/bin/python -m src.exp_neuronales realmlp

TabM_D se probo en un fold y no termino en 15 min con 4 hilos de CPU: descartado.
"""
import sys

import torch

from src import config as C
from src.evaluar import evaluar

torch.set_num_threads(4)


def _prep(x):
    x = x.drop(columns=[C.TARGET, C.MONTH], errors="ignore").copy()
    for c in x.columns:
        if str(x[c].dtype) != "category":
            x[c] = x[c].astype(float).fillna(0)
    return x


def realmlp_fp(tr, va, seed):
    from pytabkit import RealMLP_TD_Classifier
    m = RealMLP_TD_Classifier(random_state=seed, n_threads=4, device="cpu", verbosity=0)
    m.fit(_prep(tr), tr[C.TARGET].values)
    return m.predict_proba(_prep(va))[:, 1]


if __name__ == "__main__":
    if "realmlp" in sys.argv[1:]:
        evaluar("nn_realmlp", realmlp_fp, seeds=[42], nota="pytabkit RealMLP_TD defaults, CPU 4 hilos")
