import pandas as pd
from src import config as C

def load_all():
    """Une train y test para construir variables con historial común."""
    tr = pd.read_csv(C.TRAIN); tr["_origen"] = "train"
    te = pd.read_csv(C.TEST);  te["_origen"] = "test"
    df = pd.concat([tr, te], ignore_index=True)
    for c in C.BOOL_COLS:
        df[c] = df[c].astype(int)
    for c in C.CAT_COLS:
        df[c] = df[c].astype("category")
    return df
