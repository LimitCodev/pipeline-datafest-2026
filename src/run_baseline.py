"""Experimentos del pipeline. Cada experimento = un cambio contra el baseline congelado.

Uso:
  python -m src.run_baseline --out baseline_ref.json --seeds 42 43 44 45 46
  python -m src.run_baseline --variante ene_nan --ref baseline_ref.json --out resultados/ene_nan.json
  python -m src.run_baseline --ref baseline_ref.json --out resultados/x.json --folds 202610

Guards:
  * --out es obligatorio para guardar (nada se guarda si no se pide).
  * baseline_ref.json y sub_baseline.csv jamas se sobrescriben.
  * --ref y --out nunca pueden ser el mismo archivo.
"""
import argparse
import os
import time

from src import experiment as E
from src.data import load_all
from src.variantes import resolve


def main() -> None:
    ap = argparse.ArgumentParser(description="Experimentos con validacion temporal")
    ap.add_argument("--variante", default="base", help="nombre de la variante (src/variantes.py)")
    ap.add_argument("--ref", default=None, help="JSON congelado del baseline para comparar en pareado")
    ap.add_argument("--out", default=None, help="donde guardar los resultados (si no se pasa, no guarda)")
    ap.add_argument("--folds", nargs="+", type=int, default=E.FOLDS_SEL,
                    help="meses de validacion (por defecto los folds de seleccion)")
    ap.add_argument("--seeds", nargs="+", type=int, default=E.SEEDS_EXP)
    ap.add_argument("--force", action="store_true", help="permite pisar un --out existente (no las referencias)")
    args = ap.parse_args()

    spec = resolve(args.variante)
    base = os.path.basename(args.out)
    if os.path.exists(args.out) and base in E.PROTEGIDOS:
        raise SystemExit(f"ERROR: {args.out} es una referencia; usa otro --out.")
    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f"ERROR: {args.out} ya existe; usa --force u otro --out.")
    if args.ref and os.path.abspath(args.ref) == os.path.abspath(args.out or ""):
        raise SystemExit("ERROR: --ref y --out no pueden ser el mismo archivo.")

    t0 = time.time()
    df = load_all()
    res = E.run_cv(df, args.folds, args.seeds,
                   params=spec.get("params"), transformar=spec.get("transformar"),
                   filtro_train=spec.get("filtro_train"))
    dt = time.time() - t0

    res_act = E.resumen(res)
    ref = E.cargar(args.ref) if args.ref else None
    cmp_ = E.comparar(ref, res) if ref else {}
    E.imprimir(args.ref or "(sin ref)", args.variante, cmp_,
               E.resumen(ref) if ref else None, res_act)

    if args.out:
        E.guardar(args.out, experimento=args.variante, seeds=args.seeds,
                  meses=args.folds, params=E.PARAMS_BASE, resultados=res,
                  force=args.force)
        print("\nguardado en", args.out)
    print(f"tiempo: {dt:.1f}s  ({len(args.seeds)} semillas x {len(args.folds)} folds)")


if __name__ == "__main__":
    main()
