import sys
import pandas as pd
from src import config as C

ruta = sys.argv[1]
sub = pd.read_csv(ruta)
ref = pd.read_csv(C.DATA_DIR / "sample_submission.csv")
te = pd.read_csv(C.TEST)

assert list(sub.columns) == ["id_cliente", "prediccion"], "columnas incorrectas"
assert len(sub) == len(te) == len(ref), "número de filas incorrecto"
assert (sub["id_cliente"].values == te["id_cliente"].values).all(), "orden de id_cliente distinto al de test.csv"
assert sub["prediccion"].notna().all(), "hay valores vacíos"
assert sub["prediccion"].between(0, 1).all(), "probabilidades fuera de [0, 1]"
print("Entrega válida:", len(sub), "filas")
