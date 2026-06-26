# Analisis de Inteligencia Artificial - Manordomo

1. **`dataset_gestos.csv`**: Un conjunto de datos con 300 muestras correspondientes a 3 gestos diferentes (Puno, Palma Abierta, Paz). Contiene 63 variables que representan las coordenadas tridimensionales (X, Y, Z) de los 21 puntos (landmarks) de la mano generados a traves del mapeo espacial.
2. **`generar_csv.py`**: Script de Python responsable de generar la geometria 3D de las manos para crear las coordenadas simuladas y realistas del dataset, incorporando ruido estadistico natural.
3. **`modelo_no_supervisado.py`**: El script principal que procesa los datos del CSV.
4. **`grafica_clusters.png`**: Representacion visual de los resultados generados por el modelo.

## Algoritmos Utilizados

*   **Reduccion de Dimensionalidad (PCA):** Se utiliza el Analisis de Componentes Principales para reducir las 63 variables espaciales de la mano a solo 2 dimensiones principales. Esto permite simplificar la complejidad de los datos, eliminando variables redundantes, y posibilitando la visualizacion en un plano 2D conservando la mayor parte de la varianza.
*   **Agrupacion (K-Means):** Se aplica este algoritmo de aprendizaje no supervisado sobre las componentes principales obtenidas. El objetivo de K-Means en este contexto es descubrir de manera autonoma la agrupacion natural de los gestos basado unicamente en su similitud matematica, sin conocer de antemano a que gesto pertenece cada registro.

## Como compilar y ejecutar el proyecto localmente

Para ejecutar los scripts garantizando que el entorno no contamine otras instalaciones de Python, se recomienda utilizar un entorno virtual siguiendo estos pasos en PowerShell:

1. **Crear el entorno virtual**
```powershell
py -m venv venv
```

2. **Activar el entorno virtual**
```powershell
.\venv\Scripts\Activate.ps1
```

3. **Instalar las dependencias necesarias**
```powershell
pip install -r requirements.txt
```

4. **Ejecutar la validacion del modelo**
```powershell
python modelo_no_supervisado.py
```
