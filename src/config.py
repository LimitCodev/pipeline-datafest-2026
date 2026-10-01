from pathlib import Path

DATA_DIR = Path.home() / "datafest-datos"
TRAIN = DATA_DIR / "train.csv"
TEST = DATA_DIR / "test.csv"

SEED = 42
ID, MONTH, TARGET = "id_cliente", "mes", "objetivo"

CAT_COLS = ["ocupacion", "region", "canal_adquisicion",
            "banda_riesgo", "dispositivo_principal"]
BOOL_COLS = ["tiene_tarjeta_credito", "activo_movil", "es_nuevo_cliente",
             "tiene_prestamo", "tiene_seguro"]
