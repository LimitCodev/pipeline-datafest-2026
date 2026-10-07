# Correr TabICL y TabM en Google Colab (GPU)

En la laptop (CPU, 16 hilos) TabICL no terminó ni un mes en 20 minutos. En Colab con GPU es mucho más rápido.

1. En Colab: **Entorno de ejecución → Cambiar tipo de entorno → GPU T4**.
2. Clonar la rama e instalar:
   ```bash
   !git clone -b worktree-nn-retry https://github.com/LimitCodev/pipeline-datafest-2026.git
   %cd pipeline-datafest-2026
   !pip install -q pytabkit tabicl lightgbm
   ```
3. Crear `datafest-datos/` y subir `train.csv` y `test.csv` (no están en el repositorio).
4. Correr **uno por uno**:
   ```bash
   !python -m src.exp_neuronales tabicl
   !python -m src.exp_neuronales tabm
   ```
5. Descargar `resultados/nn_tabicl.json`, `resultados/nn_tabm.json` y `resultados/registro.csv` y compararlos con el modelo final (Gini 0.2636).
