"""Genera la submission de diciembre con el modelo entrenado en ene-nov.

Uso:
  python -m src.make_submission --out submissions/sub_final.csv --variante base

Guards: nunca sobrescribe submissions/sub_baseline.csv (referencia) y nunca
guarda nada sin --out explicito.
"""
import argparse
import os
import sys

import lightgbm as lgb
import pandas as pd

from src import config as C
from src import experiment as E
from src.data import load_all
from src.variantes import resolve


def main() -> None:
    ap = argparse.ArgumentParser(description="Submission de diciembre")
    ap.add_argument("--out", required=True, help="ruta del csv a generar")
    ap.add_argument("--variante", default="base")
    ap.add_argument("--seeds", nargs="+", type=int, default=[C.SEED],
                    help="promedia las predicciones de varias semillas")
    ap.add_argument("--folds-para-n-estimators", nargs="*", type=int, default=None,
                    help="si se pasa, fija n_estimators con early stopping temporal "
                         "en esos meses y luego reentrena con todo ene-nov")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    base = os.path.basename(args.out)
    if os.path.exists(args.out) and base == "sub_baseline.csv":
        raise SystemExit("ERROR: sub_baseline.csv es una referencia; usa otro --out.")
    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f"ERROR: {args.out} ya existe; usa --force u otro --out.")

    spec = resolve(args.variante)
    df = load_all()
    df2 = spec["transformar"](df.copy()) if spec.get("transformar") else E.add_features(df.copy())
    train = df2[df2["_origen"] == "train"]
    if spec.get("filtro_train"):
        train = train[spec["filtro_train"](train)]
    test = df2[df2["_origen"] == "test"]
    feats = E.columnas_features(df2)
    params = dict(E.PARAMS_BASE)
    if spec.get("params"):
        params.update(spec["params"])

    n_est = params["n_estimators"]
    if args.folds_para_n_estimators:
        mejores = []
        for m in args.folds_para_n_estimators:
            tr = train[train[C.MONTH] < m]
            va = train[train[C.MONTH] == m]
            p = dict(params)
            p.update(n_estimators=3000, learning_rate=0.03)
            m1 = lgb.LGBMClassifier(**p)
            m1.fit(tr[feats], tr[C.TARGET], eval_set=[(va[feats], va[C.TARGET])],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
            mejores.append(m1.best_iteration_)
            print(f"  early stopping en {m}: mejor iteracion = {m1.best_iteration_}")
        n_est = int(round(sum(mejores) / len(mejores)))
        print(f"n_estimators promedio = {n_est}")
        params["n_estimators"] = n_est
        params.pop("learning_rate", None)
        params["learning_rate"] = 0.03

    p = 0
    for s in args.seeds:
        model = lgb.LGBMClassifier(**{**params, "random_state": s})
        model.fit(train[feats], train[C.TARGET])
        p = p + model.predict_proba(test[feats])[:, 1] / len(args.seeds)

    out = pd.DataFrame({C.ID: test[C.ID].values, "prediccion": p})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False, lineterminator="\r\n")  # mismo formato que sample_submission.csv (CRLF)
    print("Listo:", out.shape, "->", args.out)
    if not (out["prediccion"].between(0, 1).all() and out["prediccion"].notna().all()):
        sys.exit("ERROR: predicciones invalidas")


if __name__ == "__main__":
    main()
