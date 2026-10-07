"""Sanity del evaluador y re-registro de variantes LightGBM existentes.

python -m src.eval_lgbm limpio_reg   -> debe dar delta 0 exacto (control)
"""
import sys

import lightgbm as lgb

from src import config as C
from src import experiment as E
from src.evaluar import evaluar
from src.variantes import resolve


def lgbm_fp(params):
    def fp(tr, va, seed):
        f = [c for c in tr.columns if c not in (C.TARGET, C.MONTH)]
        m = lgb.LGBMClassifier(**{**E.PARAMS_BASE, **params, "random_state": seed})
        return m.fit(tr[f], tr[C.TARGET]).predict_proba(va[f])[:, 1]
    return fp


if __name__ == "__main__":
    for v in sys.argv[1:]:
        spec = resolve(v)
        evaluar(f"lgb_{v}", lgbm_fp(spec.get("params") or {}),
                transformar=spec.get("transformar"), filtro_train=spec.get("filtro_train"),
                nota=f"variante {v} de src/variantes.py")
