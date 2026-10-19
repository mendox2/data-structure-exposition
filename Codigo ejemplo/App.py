import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

#! PRIMERA VERSIÓN DE LA FASE FINAL, BAJO REVISIÓN

print("---------------------------------------------------------")
print("DEMOSTRACION: MATRICES DENSAS VS DISPERSAS (CSR)")
print("DATASET REAL: MOVIELENS")
print("---------------------------------------------------------")

#* Rutas de los equipos
rutaIsaak = r"C:\Users\USER\OneDrive\Escritorio\data-structure-exposition\Codigo ejemplo\ratings.csv"
rutaSebastian = r"C:\Users\ACER\Desktop\data-structure-exposition\Codigo ejemplo\ratings.csv"

try:
    datos = pd.read_csv(rutaIsaak)
except FileNotFoundError:
        print(f"Se esperaba en: {rutaIsaak}")
        print(f"o en: {rutaSebastian}")
        print("Verifica que la ruta sea correcta.")
        sys.exit(1)
except Exception as e:
    print(f"ERROR: No se pudo abrir el archivo ratings.csv: {e}")
    sys.exit(1)

columnasNecesarias = {"userId", "movieId", "rating"}

if not columnasNecesarias.issubset(datos.columns):
    print("ERROR: El CSV no contiene las columnas necesarias.")
    print("Se requieren: userId, movieId y rating.")
    sys.exit(1)

#* Se limpian los datos
datos = datos.dropna(
    subset=["userId", "movieId", "rating"]
).copy()

datos["rating"] = pd.to_numeric(
    datos["rating"], errors="coerce"
)

datos = datos.dropna(subset=["rating"])

#* El dataset contiene calificaciones entre 0.5 y 5
datos = datos[
    (datos["rating"] > 0) &
    (datos["rating"] <= 5)
]

#! Revisar
datos = datos.groupby(
    ["userId", "movieId"], as_index=False
)["rating"].mean()

if datos.empty:
    print("ERROR: El dataset no contiene calificaciones validas.")
    sys.exit(1)

print("\nDATASET CARGADO CORRECTAMENTE")
print(f"Registros de calificaciones: {len(datos):,}")

#* Se convierte a matriz
indicesUsuarios, usuarios = pd.factorize(
    datos["userId"], sort=True
)

indicesPeliculas, peliculas = pd.factorize(
    datos["movieId"], sort=True
)

filas = len(usuarios)
columnas = len(peliculas)

#* Se crea la matriz "dispersa"
valores = datos["rating"].to_numpy(dtype=np.float64)

matrizDispersa = csr_matrix(
    (valores, (indicesUsuarios, indicesPeliculas)),
    shape=(filas, columnas)
)

#* Se eliminan los ceros
matrizDispersa.eliminate_zeros()

elementosNoNulos = matrizDispersa.nnz
totalPosiciones = filas * columnas

print("\nINFORMACION DE LA MATRIZ")
print(f"Usuarios (filas): {filas:,}")
print(f"Peliculas (columnas): {columnas:,}")
print(f"Dimensiones: {filas} x {columnas}")
print(f"Posiciones totales: {totalPosiciones:,}")
print(f"Elementos no nulos: {elementosNoNulos:,}")

densidad = (elementosNoNulos / totalPosiciones) * 100
print(f"Densidad de la matriz: {densidad:.4f}%")
print(f"Porcentaje de ceros: {100 - densidad:.4f}%")

#* Se crea un array de la matriz dispersa
matrizDensa = matrizDispersa.toarray()

bytesMatrizDensa = matrizDensa.nbytes
mbMatrizDensa = bytesMatrizDensa / (1024 * 1024)

print("\nREPRESENTACION DENSA (NumPy)")
print(f"Dimensiones: {filas} x {columnas}")
print(f"Memoria consumida: {mbMatrizDensa:.4f} MB")
print(f"Memoria en bytes: {bytesMatrizDensa:,}")

bytesData = matrizDispersa.data.nbytes
bytesIndices = matrizDispersa.indices.nbytes
bytesIndptr = matrizDispersa.indptr.nbytes

bytesMatrizDispersa = (
    bytesData + bytesIndices + bytesIndptr
)

mbMatrizDispersa = bytesMatrizDispersa / (1024 * 1024)

print("\nREPRESENTACION DISPERSA COMPRIMIDA (CSR)")
print(f"Memoria de data: {bytesData:,} bytes")
print(f"Memoria de indices: {bytesIndices:,} bytes")
print(f"Memoria de indptr: {bytesIndptr:,} bytes")
print(f"Memoria total: {mbMatrizDispersa:.4f} MB")
print(f"Memoria total en bytes: {bytesMatrizDispersa:,}")

#* Se calcula el ahorro de memoria
ahorroPorcentaje = (
    (bytesMatrizDensa - bytesMatrizDispersa)
    / bytesMatrizDensa
) * 100

print("\nRESULTADO DE EFICIENCIA")
print(
    f"Ahorro de memoria: {ahorroPorcentaje:.2f}% "
    "menos que la representacion densa"
)

#* Se multiplica la matriz y el vector
vectorN = np.ones(columnas, dtype=np.float64)

resultado = matrizDispersa.dot(vectorN)

print("\nOPERACION MATEMATICA (SpMV)")
print("Multiplicacion completada usando CSR.")
print("Primeros 3 resultados del vector:")

print(resultado[:3])

resultadoDenso = matrizDensa.dot(vectorN)

print(
    "Los resultados coinciden:",
    np.allclose(resultado, resultadoDenso)
)

print("---------------------------------------------------------")
print("PROCESO FINALIZADO")
print("---------------------------------------------------------")
