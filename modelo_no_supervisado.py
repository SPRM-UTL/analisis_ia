import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Generar Dataset Simulado de Manordomo (Puntos de la mano de MediaPipe)
# MediaPipe Hand Landmarks: 21 puntos (x, y, z) = 63 características.
print("Generando datos simulados de 3 gestos (Puño, Palma Abierta, Paz)...")
np.random.seed(42)
num_samples_per_gesture = 100

# Gesto 1: Puño (Dedos cerrados)
fist_data = np.random.normal(loc=0.2, scale=0.05, size=(num_samples_per_gesture, 63))
# Gesto 2: Palma Abierta (Dedos extendidos)
palm_data = np.random.normal(loc=0.8, scale=0.05, size=(num_samples_per_gesture, 63))
# Gesto 3: Paz (Índice y medio extendidos)
peace_data = np.random.normal(loc=0.5, scale=0.05, size=(num_samples_per_gesture, 63))

# Unir todo el dataset (Total: 300 muestras, 63 columnas)
X_raw = np.vstack((fist_data, palm_data, peace_data))

# Etiquetas REALES solo para colorear después y comprobar si el modelo adivinó bien (El modelo NO las verá)
y_real = np.array([0]*100 + [1]*100 + [2]*100)
gestures = {0: 'Puño', 1: 'Palma Abierta', 2: 'Paz'}

print(f"Dimensiones del dataset original: {X_raw.shape}")

# 2. Reducción de Dimensionalidad (PCA)
print("\n--- APLICANDO REDUCCIÓN DE DIMENSIONALIDAD (PCA) ---")
# Reducimos de 63 dimensiones (puntos) a solo 2 componentes principales (X, Y) para poder graficarlos
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_raw)
print(f"Dimensiones después de PCA: {X_pca.shape}")
print(f"Varianza explicada por las 2 componentes: {sum(pca.explained_variance_ratio_):.2%}")


# 3. Agrupación (Clustering No Supervisado - K-Means)
print("\n--- APLICANDO AGRUPACIÓN (K-MEANS) ---")
# Le pedimos al algoritmo que agrupe en 3 clusters, SIN decirle qué dato es qué gesto
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_pca) # Adivina a qué grupo pertenece cada punto
print("K-Means ha agrupado los datos exitosamente.")


# 4. Graficar los resultados
plt.figure(figsize=(14, 6))

# Gráfica 1: Cómo lo agrupó el algoritmo K-Means (Aprendizaje No Supervisado)
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', edgecolors='k', s=50)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.7, marker='X', label='Centroides')
plt.title('Agrupación NO SUPERVISADA (K-Means)\nEl modelo encontró estos 3 grupos por sí solo')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Gráfica 2: Realidad (Etiquetas reales ocultas)
plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_real, cmap='plasma', edgecolors='k', s=50)
plt.title('Realidad Oculta (Gestos verdaderos)\nComparamos para ver si el modelo acertó')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True, linestyle='--', alpha=0.6)

# Añadir leyenda de colores reales
handles, _ = scatter2.legend_elements()
plt.legend(handles, ['Puño', 'Palma', 'Paz'])

plt.tight_layout()
plt.savefig('grafica_clusters.png')
print("\nSe ha generado y guardado la gráfica 'grafica_clusters.png'.")
print("Análisis completado. Listo para entregar en el repositorio.")
