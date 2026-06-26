import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Cargar Dataset Real (Puntos de la mano de MediaPipe)
# El CSV contiene 63 características (21 puntos con x, y, z)
print("Cargando datos desde dataset_gestos.csv...")
df = pd.read_csv("dataset_gestos.csv")

# Separar las características (X) y la etiqueta real (y_real)
X_raw = df.drop(columns=["etiqueta_real"]).values
y_labels = df["etiqueta_real"].values

# Convertir las etiquetas de texto a números para graficarlas después (solo para validación visual)
# El modelo K-Means NUNCA verá estas etiquetas al agrupar
label_map = {'Puño': 0, 'Palma Abierta': 1, 'Paz': 2}
y_real = np.array([label_map[label] for label in y_labels])

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
