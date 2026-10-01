from src import config as C

NUM_LAG = ["saldo_promedio", "visitas_web_ultimos_90_dias",
           "dias_ultima_transaccion", "numero_productos",
           "dias_ultima_interaccion"]

def add_features(df):
    """Solo usa filas de meses ANTERIORES del mismo cliente (sin fuga)."""
    g = df.sort_values([C.ID, C.MONTH]).groupby(C.ID)
    df["meses_previos"] = g.cumcount()
    df["tiene_historial"] = (df["meses_previos"] > 0).astype(int)
    for c in NUM_LAG:
        df[f"{c}_diff1"] = g[c].diff()  # cambio vs. observación anterior
    return df
