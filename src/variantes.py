"""Registro de variantes: un experimento = un cambio.

Cada variante puede traer:
  transformar(df)   -> cambia/features (se aplica sobre una copia, antes de validar)
  filtro_train(df)  -> quita filas del ENTRENAMIENTO (la validacion no se toca)
  params            -> dict que sobreescribe PARAMS_BASE

Las features siempre se construyen con src.features.add_features (sin fuga) y
solo se le suman cambios declarados aqui.
"""
from src.features import add_features

# ---------------------------------------------------------------------------
# E2 - enero censurado: en 202601 todos los clientes tienen meses_previos=0 y
# tiene_historial=0 por construccion (la ventana de datos empieza en enero), asi
# que esas dos columnas dejan de significar "sin historial" y pasan a ser un
# indicador de "es enero". En diciembre no hay filas de enero: se le pone NaN.
# ---------------------------------------------------------------------------
def _ene_nan(df):
    df = add_features(df.copy())
    ene = df["mes"] == 202601
    df.loc[ene, "meses_previos"] = float("nan")
    df.loc[ene, "tiene_historial"] = float("nan")
    return df


# Alternativa: excluir enero del entrenamiento (no de la validacion).
def _filtro_sin_ene(df):
    return df["mes"] != 202601


VARIANTES = {
    "base": {},
    "ene_nan": {"transformar": _ene_nan},
    "sin_ene": {"filtro_train": _filtro_sin_ene},
}


def resolve(nombre: str) -> dict:
    if nombre not in VARIANTES:
        raise SystemExit(f"variante desconocida: {nombre!r}. "
                         f"Disponibles: {', '.join(sorted(VARIANTES))}")
    return VARIANTES[nombre]


# ---------------------------------------------------------------------------
# Mejoras 2026-10-07. Hallazgo EDA: todas las columnas son estaticas por cliente
# salvo dias_ultima_interaccion, asi que los diff1 de columnas estaticas son
# siempre 0/NaN (ruido). Se quitan y se agrega historial de interacciones.
# Todo usa solo filas del mismo cliente con mes <= m (sin fuga).
# ---------------------------------------------------------------------------
PARAMS_REG = dict(learning_rate=0.01, n_estimators=1000, num_leaves=7,
                  min_child_samples=300, colsample_bytree=0.5, reg_lambda=10)

ESTATICOS_DIFF = ["saldo_promedio_diff1", "visitas_web_ultimos_90_dias_diff1",
                  "dias_ultima_transaccion_diff1", "numero_productos_diff1"]


def _limpio(df):
    return add_features(df).drop(columns=ESTATICOS_DIFF)


def _interacciones(df):
    df = _limpio(df).sort_values(["id_cliente", "mes"])
    I, T = df["dias_ultima_interaccion"], df["dias_ultima_transaccion"]
    g = df.groupby("id_cliente")["dias_ultima_interaccion"]
    df["gap_tx_int"] = T - I                     # >0: hubo contacto despues de la ultima transaccion
    df["int_igual_tx"] = (I == T).astype(int)
    prev = g.shift()
    df["int_prev"] = prev
    df["int_min_prev"] = prev.groupby(df["id_cliente"]).cummin()
    df["int_media_prev"] = prev.groupby(df["id_cliente"]).transform(lambda s: s.expanding().mean())
    nueva = (I < prev).astype(float).where(prev.notna())   # bajo el contador = contacto nuevo
    df["n_contactos"] = nueva.groupby(df["id_cliente"]).cumsum()
    df["int_min_hist"] = g.cummin()
    return df.sort_index()


VARIANTES.update({
    "reg": {"params": PARAMS_REG},
    "limpio_reg": {"transformar": _limpio, "params": PARAMS_REG},
    "inter_reg": {"transformar": _interacciones, "params": PARAMS_REG},
})


# Reglas vistas en el EDA del 2026-10-07 (tablas cruzadas de tasa de conversion):
# low + >=3 productos ~0.29 vs ~0.14; high + transaccion >180 dias ~0.05 vs ~0.15;
# tarjeta + movil activos sube, mas aun en low con >=2 productos. Los arboles
# encuentran estos cortes solos, pero con un flag lo hacen en un split.
def _reglas(df):
    df = _limpio(df)
    low, high = df["banda_riesgo"] == "low", df["banda_riesgo"] == "high"
    tx, prod = df["dias_ultima_transaccion"], df["numero_productos"]
    tm = (df["tiene_tarjeta_credito"] == 1) & (df["activo_movil"] == 1)
    df["r_low_prod3"] = (low & (prod >= 3)).astype(int)
    df["r_high_tx180"] = (high & (tx > 180)).astype(int)
    df["r_high_tx100"] = (high & (tx > 100)).astype(int)
    df["r_tarjeta_movil"] = tm.astype(int)
    df["r_low_tm_prod2"] = (low & tm & (prod >= 2)).astype(int)
    return df


VARIANTES["reglas_reg"] = {"transformar": _reglas, "params": PARAMS_REG}


# Features con ganancia ~0 y tasa plana en el EDA (ruido para el modelo).
RUIDO = ["dispositivo_principal", "tiene_prestamo", "tiene_seguro", "es_nuevo_cliente",
         "dia_preferido_pago", "visitas_web_ultimos_90_dias", "region",
         "antiguedad_direccion_meses"]


def _sin_ruido(df):
    return _limpio(df).drop(columns=RUIDO)




VARIANTES["sin_ruido_reg"] = {"transformar": _sin_ruido, "params": PARAMS_REG}
VARIANTES["reglas_sin_ene_reg"] = {"transformar": _reglas, "params": PARAMS_REG,
                                   "filtro_train": _filtro_sin_ene}
