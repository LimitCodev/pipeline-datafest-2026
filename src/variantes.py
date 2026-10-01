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
