"""Modelos tabulares neuronales recientes (pytabkit, tabicl) con el juez comun src.evaluar.

Corre en .venv-ag (Python 3.12 + torch) o en Colab con GPU, UN modelo a la vez:
    .venv-ag/bin/python -m src.exp_neuronales realmlp
    .venv-ag/bin/python -m src.exp_neuronales tabicl
    .venv-ag/bin/python -m src.exp_neuronales tabm

Sin limite de tiempo: las reglas de la competencia no lo imponen.
En CPU (laptop, 16 hilos) TabICL no termino ni un mes en 20 min: usar GPU.
"""
import os
import sys

import torch

from src import config as C
from src.evaluar import evaluar

# un modelo a la vez, con todos los hilos (NN_HILOS para cambiarlo)
HILOS = int(os.environ.get("NN_HILOS", os.cpu_count()))
torch.set_num_threads(HILOS)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU en Colab


def _prep(x):
    x = x.drop(columns=[C.TARGET, C.MONTH], errors="ignore").copy()
    for c in x.columns:
        if str(x[c].dtype) != "category":
            x[c] = x[c].astype(float).fillna(0)
    return x


def realmlp_fp(tr, va, seed):
    from pytabkit import RealMLP_TD_Classifier
    m = RealMLP_TD_Classifier(random_state=seed, n_threads=HILOS, device=DEVICE, verbosity=0)
    m.fit(_prep(tr), tr[C.TARGET].values)
    return m.predict_proba(_prep(va))[:, 1]


def tabicl_fp(tr, va, seed):
    from tabicl import TabICLClassifier
    # batch_size=1 y offload: un miembro del ensamble a la vez, cabe en 13 GB
    m = TabICLClassifier(random_state=seed, device=DEVICE, batch_size=1,
                         offload_mode="auto", n_jobs=HILOS)
    m.fit(_prep(tr), tr[C.TARGET].values)
    return m.predict_proba(_prep(va))[:, 1]


def tabm_fp(tr, va, seed):
    from pytabkit import TabM_D_Classifier
    m = TabM_D_Classifier(random_state=seed, n_threads=HILOS, device=DEVICE, verbosity=1)
    m.fit(_prep(tr), tr[C.TARGET].values)
    return m.predict_proba(_prep(va))[:, 1]


if __name__ == "__main__":
    modo = sys.argv[1:]
    if "realmlp" in modo:
        evaluar("nn_realmlp", realmlp_fp, seeds=[42], nota="pytabkit RealMLP_TD defaults, CPU 4 hilos")
    if "tabicl" in modo:
        evaluar("nn_tabicl", tabicl_fp, seeds=[42], nota=f"TabICL v2 defaults, batch_size=1, CPU {HILOS} hilos")
    if "tabm" in modo:
        evaluar("nn_tabm", tabm_fp, seeds=[42], nota=f"pytabkit TabM_D defaults, CPU {HILOS} hilos, sin limite de tiempo")
