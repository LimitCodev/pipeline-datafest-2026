"""AutoGluon Tabular juzgado por src.evaluar (1 semilla por costo).

.venv-ag/bin/python -m src.automl_autogluon medium   -> ag_medium
.venv-ag/bin/python -m src.automl_autogluon best     -> ag_best
`mes` se quita: el test es un mes futuro, como numero solo extrapolaria.
"""
import os
import shutil
import sys
import tempfile

from autogluon.tabular import TabularPredictor

from src import config as C
from src.evaluar import evaluar

SCRATCH = os.path.join(tempfile.gettempdir(), "datafest_ag")  # modelos temporales, se borran tras cada fit
LB_DIR = "resultados/autogluon"
CONFIGS = {
    "medium": {"presets": "medium_quality"},
    # dynamic_stacking off: con 240 s, DyStack se comeria la mitad del tiempo.
    "best": {"presets": "best_quality", "dynamic_stacking": False},
    # validacion interna honesta: tuning_data = ultimo mes de tr, entrena con los anteriores
    "temporal": {"presets": "medium_quality", "num_bag_folds": 0, "_temporal": True},
}


def ag_fp(nombre, cfg):
    def fp(tr, va, seed):
        mes = int(va[C.MONTH].iloc[0])
        assert tr[C.MONTH].max() < mes  # AutoGluon nunca ve el mes validado
        path = f"{SCRATCH}/{nombre}_{mes}_{seed}"
        shutil.rmtree(path, ignore_errors=True)
        cfg_ = dict(cfg)
        if cfg_.pop("_temporal", False):
            ult = tr[C.MONTH].max()
            cfg_["tuning_data"] = tr[tr[C.MONTH] == ult].drop(columns=[C.MONTH])
            tr = tr[tr[C.MONTH] < ult]
        try:
            p = TabularPredictor(label=C.TARGET, eval_metric="roc_auc", path=path,
                                 verbosity=1).fit(
                tr.drop(columns=[C.MONTH]), time_limit=240, num_cpus=4, num_gpus=0,
                memory_limit=4, **cfg_)
            p.leaderboard(silent=True).to_csv(f"{LB_DIR}/{nombre}_{mes}_s{seed}.csv", index=False)
            return p.predict_proba(va.drop(columns=[C.MONTH]))[1].to_numpy()
        finally:
            shutil.rmtree(path, ignore_errors=True)
    return fp


if __name__ == "__main__":
    os.makedirs(LB_DIR, exist_ok=True)
    for k in sys.argv[1:]:
        evaluar(f"ag_{k}", ag_fp(f"ag_{k}", CONFIGS[k]), seeds=[42],
                nota=f"AutoGluon 1.6.3 {CONFIGS[k]}, 240s/fit, 4 cpu, 4GB, sin mes, seed 42")
