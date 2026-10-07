"""Experimentos de ENCUADRE y FEATURES con LightGBM PARAMS_REG (juez: src.evaluar).

python -m src.exp_encuadre enc_w_cliente enc_te ...   (o sin args = todos)

Toda estadistica con objetivo (target encoding) se calcula dentro de fit_predict,
para cada fila solo con meses ESTRICTAMENTE anteriores (en va: todo tr, que ya es < m).
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config as C
from src import experiment as E
from src.evaluar import evaluar
from src.variantes import PARAMS_REG, _limpio

T, M = C.TARGET, C.MONTH
P = {**E.PARAMS_BASE, **PARAMS_REG}


# ---------------- transformaciones (sin objetivo: solo la fila / su pasado) ----
def _con_cid(df):
    df = _limpio(df)
    df["_cid"] = df[C.ID]            # no es feature: fit_predict la quita
    return df


def _ord_ratios(df):
    df = _limpio(df)
    df["banda_riesgo"] = df["banda_riesgo"].map({"low": 0, "medium": 1, "high": 2}).astype(float)
    df["saldo_ingresos"] = df["saldo_promedio"] / df["ingresos"]
    df["deuda"] = df["ratio_deuda_ingresos"] * df["ingresos"]
    df["saldo_producto"] = df["saldo_promedio"] / df["numero_productos"].clip(lower=1)
    return df


def _entrada(df):
    df = _limpio(df)
    idx = (df[M] // 100 - 2026) * 12 + df[M] % 100          # 202601 -> 1
    df["mes_entrada"] = idx - df["meses_previos"]
    df["entrada_censurada"] = (df["mes_entrada"] == 1).astype(int)
    return df


# ---------------- target encoding temporal ------------------------------------
COMBOS = [["banda_riesgo"], ["ocupacion"], ["canal_adquisicion"], ["region"],
          ["banda_riesgo", "numero_productos"],
          ["banda_riesgo", "tiene_tarjeta_credito", "activo_movil"],
          ["banda_riesgo", "ocupacion"], ["canal_adquisicion", "numero_productos"]]
SUAVE = 50


def _key(d, cols):
    return d[cols].astype(str).agg("|".join, axis=1)


def _te(tr, va):
    tr, va = tr.copy(), va.copy()
    for cols in COMBOS:
        name = "te_" + "_".join(cols)
        ktr, kva = _key(tr, cols), _key(va, cols)
        out = pd.Series(np.nan, index=tr.index)
        for t in sorted(tr[M].unique()):
            past = tr[M] < t
            if not past.any():
                continue                          # primer mes: sin pasado -> NaN
            y = tr.loc[past, T]
            g = y.groupby(ktr[past]).agg(["sum", "count"])
            enc = (g["sum"] + SUAVE * y.mean()) / (g["count"] + SUAVE)
            cur = tr[M] == t
            out[cur] = ktr[cur].map(enc).fillna(y.mean()).values
        tr[name] = out
        g = tr[T].groupby(ktr).agg(["sum", "count"])
        va[name] = kva.map((g["sum"] + SUAVE * tr[T].mean()) / (g["count"] + SUAVE)).fillna(tr[T].mean())
    return tr, va


# ---------------- fit_predict generico ----------------------------------------
def fp_factory(params=None, peso=None, filtro=None, prep=None, ranker=False, n_bag=1):
    def fp(tr, va, seed):
        if filtro is not None:
            tr = tr[filtro(tr)]
        if prep is not None:
            tr, va = prep(tr, va)
        w = peso(tr) if peso is not None else None
        f = [c for c in tr.columns if c not in (T, M, "_cid")]
        preds = []
        for k in range(n_bag):
            p = {**P, **(params or {}), "random_state": seed * 1000 + k}
            if ranker:
                # LightGBM limita 10000 filas por query: cada mes se parte en 2 mitades al azar
                half = np.random.default_rng(seed).integers(0, 2, len(tr))
                o = np.lexsort((half, tr[M].values))
                trs = tr.iloc[o]
                grp = pd.Series(1, index=range(len(trs))).groupby(
                    [trs[M].values, half[o]], sort=True).size().values
                m = lgb.LGBMRanker(**p).fit(trs[f], trs[T], group=grp)
                preds.append(m.predict(va[f]))
            else:
                m = lgb.LGBMClassifier(**p).fit(tr[f], tr[T], sample_weight=w)
                preds.append(m.predict_proba(va[f])[:, 1])
        return np.mean(preds, 0)
    return fp


def _w_cliente(tr):            # cada cliente pesa lo mismo, sin importar cuantos meses sobrevivio
    return (1.0 / tr.groupby("_cid")[T].transform("size")).values * len(tr) / tr["_cid"].nunique()


def _ultima(tr):               # ultima fila conocida de cada cliente en el train del fold
    return tr[M] == tr.groupby("_cid")[M].transform("max")


MONO = ["numero_productos", "dias_ultima_transaccion"]


def _mono_fp(tr, va, seed):
    f = [c for c in tr.columns if c not in (T, M)]
    mc = [1 if c == MONO[0] else -1 if c == MONO[1] else 0 for c in f]
    return fp_factory({"monotone_constraints": mc})(tr, va, seed)


EXPS = {
    "enc_w_cliente": dict(fp=fp_factory(peso=_w_cliente), transformar=_con_cid,
                          nota="peso 1/(n filas del cliente en train): corrige supervivientes"),
    "enc_ultima": dict(fp=fp_factory(filtro=_ultima), transformar=_con_cid,
                       nota="solo ultima fila por cliente en train"),
    "enc_te": dict(fp=fp_factory(prep=_te), transformar=None,
                   nota="target encoding temporal (meses < t) de 4 cat + 4 combos, suave 50"),
    "enc_rank": dict(fp=fp_factory({"objective": "rank_xendcg"}, ranker=True), transformar=None,
                     nota="LGBMRanker rank_xendcg agrupado por mes"),
    "enc_mono": dict(fp=_mono_fp, transformar=None,
                     nota="monotone numero_productos +, dias_ultima_transaccion -"),
    "enc_ord_ratios": dict(fp=fp_factory(), transformar=_ord_ratios,
                           nota="banda_riesgo ordinal + saldo/ingresos, deuda, saldo/producto"),
    "enc_entrada": dict(fp=fp_factory(), transformar=_entrada,
                        nota="mes de entrada del cliente + flag entrada censurada (enero)"),
    "enc_bag": dict(fp=fp_factory({"subsample": 0.7}, n_bag=5), transformar=None,
                    nota="bagging 5 LGBM (semilla distinta, subsample 0.7)"),
}


if __name__ == "__main__":
    for n in sys.argv[1:] or list(EXPS):
        e = EXPS[n]
        evaluar(n, e["fp"], transformar=e["transformar"], nota=e["nota"])
