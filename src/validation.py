from sklearn.metrics import roc_auc_score
from src import config as C

def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1

def folds(train, meses_val):
    """Entrena con meses < m, valida en el mes m."""
    for m in meses_val:
        yield m, train[train[C.MONTH] < m], train[train[C.MONTH] == m]

def reporte(va, p):
    out = {"global": gini(va[C.TARGET], p)}
    for nombre, mask in [("nuevos", va["tiene_historial"] == 0),
                         ("historial", va["tiene_historial"] == 1)]:
        y = va.loc[mask, C.TARGET]
        if y.nunique() == 2:
            out[nombre] = gini(y, p[mask.values])
    return {k: round(v, 4) for k, v in out.items()}
