"""Harness de experimentos: validación temporal, semillas pareadas y comparación.

REGLA DE MEDICION: gini(), folds() y reporte() viven en src/validation.py y NO se
tocan desde aqui (ni desde ningun lado) sin permiso. Este modulo solo los llama.

Toda corrida valida con meses < m y evalua en m (validation.folds). Las features se
construyen una vez por corrida y solo usan filas del mismo cliente con mes <= m
(src.features.add_features); no hay estadisticas globales aqui.
"""
from __future__ import annotations

import json
import os
from typing import Callable

import lightgbm as lgb

from src import config as C
from src.features import add_features
from src.validation import folds, reporte

# Folds de SELECCION (202611 esta sellado: solo para confirmar finalistas).
FOLDS_SEL = [202607, 202608, 202609, 202610]
FOLD_SELLADO = 202611
SEEDS_RUIDO = [42, 43, 44, 45, 46]   # ruido de semilla del baseline
SEEDS_EXP = [42, 43, 44]             # semillas pareadas para experimentos

PARAMS_BASE = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    random_state=C.SEED,
    n_jobs=4,  # no saturar la laptop (16 hilos, 13 GB)
    verbose=-1,
)

# Archivos de salida que NUNCA se sobrescriben (referencias congeladas).
PROTEGIDOS = ("baseline_ref.json", "sub_baseline.csv")

CLAVES = ("global", "nuevos", "historial")


def columnas_features(df) -> list[str]:
    return [c for c in df.columns if c not in (C.ID, C.MONTH, C.TARGET, "_origen")]


def run_cv(df, meses, seeds, *, params=None,
           transformar: Callable | None = None,
           filtro_train: Callable | None = None) -> dict[int, dict[int, dict]]:
    """Entrena con meses < m y valida en m. Devuelve {seed: {mes: reporte}}."""
    df2 = transformar(df.copy()) if transformar else add_features(df.copy())
    train_all = df2[df2["_origen"] == "train"]
    if filtro_train is not None:
        train_all = train_all[filtro_train(train_all)]
    feats = columnas_features(df2)
    p = dict(PARAMS_BASE)
    if params:
        p.update(params)
    out: dict[int, dict[int, dict]] = {}
    for s in seeds:
        p["random_state"] = s
        for m, tr, va in folds(train_all, meses):
            model = lgb.LGBMClassifier(**p)
            model.fit(tr[feats], tr[C.TARGET])
            pred = model.predict_proba(va[feats])[:, 1]
            out.setdefault(s, {})[m] = reporte(va, pred)
    return out


def guardar(path: str, *, experimento: str, seeds, meses, params,
            resultados: dict, force: bool = False) -> None:
    base = os.path.basename(path)
    if os.path.exists(path) and base in PROTEGIDOS:
        raise SystemExit(
            f"ERROR: {path} es una referencia congelada y no se sobrescribe.\n"
            f"       Usa otro nombre (p. ej. baseline_ref_v2.json).")
    if os.path.exists(path) and not force:
        raise SystemExit(
            f"ERROR: {path} ya existe. Usa --force para sobrescribir u otro --out.")
    payload = {
        "experimento": experimento,
        "seeds": list(seeds),
        "meses": [int(m) for m in meses],
        "params": {k: v for k, v in params.items() if k != "random_state"},
        "resultados": {str(s): {str(m): r for m, r in sorted(rs.items())}
                       for s, rs in sorted(resultados.items())},
    }
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)


def cargar(path: str) -> dict[int, dict[int, dict]]:
    with open(path) as f:
        payload = json.load(f)
    return {int(s): {int(m): r for m, r in rs.items()}
            for s, rs in payload["resultados"].items()}


def media(xs) -> float:
    return sum(xs) / len(xs)


def sd(xs) -> float:
    return 0.0 if len(xs) < 2 else (sum((x - media(xs)) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def resumen(resultados: dict[int, dict[int, dict]]) -> dict:
    """Medias por (mes, clave) sobre semillas y media global sobre todo."""
    out = {"por_mes": {}, "media": {}, "sd_seeds": {}}
    for m in sorted({m for rs in resultados.values() for m in rs}):
        out["por_mes"][m] = {}
        for k in CLAVES:
            xs = [rs[m][k] for rs in resultados.values() if m in rs and k in rs[m]]
            if xs:
                out["por_mes"][m][k] = round(media(xs), 4)
    for k in CLAVES:
        xs = [r[k] for rs in resultados.values() for m, r in rs.items() if k in r]
        if xs:
            out["media"][k] = round(media(xs), 4)
    # ruido de semilla: sd entre semillas del Gini global, promediada por fold
    ruido = {}
    for k in CLAVES:
        sds = []
        for m in sorted({m for rs in resultados.values() for m in rs}):
            xs = [rs[m][k] for rs in resultados.values() if m in rs and k in rs[m]]
            if len(xs) >= 2:
                sds.append(sd(xs))
        if sds:
            ruido[k] = round(media(sds), 4)
    out["sd_seeds"] = ruido
    return out


def comparar(ref: dict[int, dict[int, dict]], act: dict[int, dict[int, dict]]) -> dict:
    """Delta pareado por (semilla, fold) contra la referencia congelada."""
    out = {}
    for k in CLAVES:
        pares = []
        for s in sorted(set(ref) & set(act)):
            for m in sorted(set(ref[s]) & set(act[s])):
                if k in ref[s][m] and k in act[s][m]:
                    pares.append((s, m, act[s][m][k] - ref[s][m][k]))
        if not pares:
            continue
        por_fold: dict[int, list[float]] = {}
        for _, m, d in pares:
            por_fold.setdefault(m, []).append(d)
        fold_media = {m: round(media(v), 4) for m, v in sorted(por_fold.items())}
        deltas = [d for _, _, d in pares]
        out[k] = {
            "delta_media": round(media(deltas), 4),
            "delta_sd": round(sd(deltas), 4),
            "n_pares": len(deltas),
            "delta_por_fold": fold_media,
            "mejora_folds": f"{sum(1 for v in fold_media.values() if v > 0)}/{len(fold_media)}",
        }
    return out


def imprimir(nombre_ref: str, nombre_act: str, cmp_: dict, res_ref=None, res_act=None) -> None:
    if res_ref is not None:
        print(f"\n[{nombre_ref}] Gini por fold (media de semillas):")
        for m, d in res_ref["por_mes"].items():
            print(f"  {m} {d}")
        print(f"  media={res_ref['media']}  sd_seeds={res_ref['sd_seeds']}")
    if res_act is not None:
        print(f"\n[{nombre_act}] Gini por fold (media de semillas):")
        for m, d in res_act["por_mes"].items():
            print(f"  {m} {d}")
        print(f"  media={res_act['media']}  sd_seeds={res_act['sd_seeds']}")
    if cmp_:
        print(f"\n[pareado] {nombre_act} - {nombre_ref}")
        for k, d in cmp_.items():
            print(f"  {k}: Δ={d['delta_media']:+.4f} ± {d['delta_sd']:.4f} "
                  f"(n={d['n_pares']}) mejora en {d['mejora_folds']} folds")
            print(f"      por fold: {d['delta_por_fold']}")
