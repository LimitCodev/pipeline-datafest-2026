import os
import lightgbm as lgb
from src import config as C
from src.data import load_all
from src.features import add_features
from src.validation import folds, reporte

df = add_features(load_all())
train = df[df["_origen"] == "train"]
FEATURES = [c for c in df.columns
            if c not in (C.ID, C.MONTH, C.TARGET, "_origen")]

PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              random_state=C.SEED, n_jobs=max(1, (os.cpu_count() or 2) - 1),
              verbose=-1)

for m, tr, va in folds(train, [202609, 202610, 202611]):
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(tr[FEATURES], tr[C.TARGET])
    p = model.predict_proba(va[FEATURES])[:, 1]
    print(m, reporte(va, p))
