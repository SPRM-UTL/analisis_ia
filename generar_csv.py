import numpy as np
import pandas as pd

np.random.seed(42)
num_samples = 100

def get_base_hand():
    lm = np.zeros((21, 3))
    # Base de la mano y muñeca
    lm[0] = [0.5, 0.9, 0.0]
    lm[1] = [0.4, 0.8, 0.0]
    lm[2] = [0.3, 0.7, 0.0]
    lm[3] = [0.2, 0.6, 0.0] 
    lm[4] = [0.1, 0.5, 0.0] 
    # Nudillos (MCP)
    lm[5] = [0.3, 0.6, 0.0]
    lm[9] = [0.4, 0.6, 0.0]
    lm[13] = [0.5, 0.6, 0.0]
    lm[17] = [0.6, 0.6, 0.0]
    return lm

def create_fist():
    lm = get_base_hand()
    # Dedos cerrados hacia la palma
    for mcp in [5, 9, 13, 17]:
        base = lm[mcp]
        lm[mcp+1] = base + [0, 0.1, 0]
        lm[mcp+2] = base + [0, 0.2, 0]
        lm[mcp+3] = base + [0, 0.25, -0.1]
    return lm.flatten()

def create_palm():
    lm = get_base_hand()
    # Dedos estirados (hacia arriba, y disminuye)
    for mcp in [5, 9, 13, 17]:
        base = lm[mcp]
        lm[mcp+1] = base + [0, -0.1, 0]
        lm[mcp+2] = base + [0, -0.2, 0]
        lm[mcp+3] = base + [0, -0.3, 0]
    return lm.flatten()

def create_peace():
    lm = get_base_hand()
    # Índice y medio estirados
    for mcp in [5, 9]:
        base = lm[mcp]
        lm[mcp+1] = base + [0, -0.1, 0]
        lm[mcp+2] = base + [0, -0.2, 0]
        lm[mcp+3] = base + [0, -0.3, 0]
    # Anular y meñique cerrados
    for mcp in [13, 17]:
        base = lm[mcp]
        lm[mcp+1] = base + [0, 0.1, 0]
        lm[mcp+2] = base + [0, 0.2, 0]
        lm[mcp+3] = base + [0, 0.25, -0.1]
    return lm.flatten()

data = []
labels = []
for _ in range(num_samples):
    # Agregar algo de "ruido" para simular variaciones naturales de la mano
    data.append(create_fist() + np.random.normal(0, 0.02, 63))
    labels.append("Puño")
for _ in range(num_samples):
    data.append(create_palm() + np.random.normal(0, 0.02, 63))
    labels.append("Palma Abierta")
for _ in range(num_samples):
    data.append(create_peace() + np.random.normal(0, 0.02, 63))
    labels.append("Paz")

columns = []
for i in range(21):
    columns.extend([f"landmark_{i}_x", f"landmark_{i}_y", f"landmark_{i}_z"])

df = pd.DataFrame(data, columns=columns)
df["etiqueta_real"] = labels

df.to_csv("dataset_gestos.csv", index=False)
print("dataset_gestos.csv creado exitosamente con 63 columnas de landmarks espaciales.")
