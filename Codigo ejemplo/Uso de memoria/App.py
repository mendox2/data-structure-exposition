import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

print("---------------------------------------------------------")
print("Matriz dispersa de rutas aéreas desde CSV (OpenFlights)")
print("---------------------------------------------------------")

nombresOpenFlights = [
    "airline",
    "airline_id",
    "source_airport",
    "source_airport_id",
    "destination_airport",
    "destination_airport_id",
    "codeshare",
    "stops",
    "equipment"
]

routesCSV = "Codigo ejemplo/routes.csv"

try:
    datosRutas = pd.read_csv(routesCSV, na_values=["\\N"])
    columnasNormalizadas = {
        str(col).strip().lower().replace("_", " ").replace("-", " "): col
        for col in datosRutas.columns
    }
    opcionesOrigen = (
        "source airport id",
        "source id"
    )

    opcionesDestino = (
        "destination airport id",
        "dest airport id",
        "destination id",
        "dest id"
    )

    colOrigen = next(
        (columnasNormalizadas[c] for c in opcionesOrigen if c in columnasNormalizadas),
        None,
    )
    colDestino = next(
        (columnasNormalizadas[c] for c in opcionesDestino if c in columnasNormalizadas),
        None,
    )

    if colOrigen is None or colDestino is None:
        datosRutas = pd.read_csv(
            routesCSV,
            header=None,
            names=nombresOpenFlights,
            na_values=["\\N"]
        )

        colOrigen = "source_airport_id"
        colDestino = "destination_airport_id"

except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo: {routesCSV}")
    sys.exit(1)

except Exception as e:
    print(f"ERROR al cargar las rutas aéreas: {e}")
    sys.exit(1)


rutasValidas = datosRutas[[colOrigen, colDestino]].copy()
rutasValidas.columns = ["origen", "destino"]
rutasValidas = rutasValidas.apply(pd.to_numeric, errors="coerce").dropna()
rutasValidas = rutasValidas[
    (rutasValidas["origen"] % 1 == 0) & (rutasValidas["destino"] % 1 == 0)
].astype(np.int64).drop_duplicates()

if rutasValidas.empty:
    print("ERROR: No se encontraron rutas aéreas válidas.")
    sys.exit(1)

print("\nCSV Cargado correctamente")
print(f"Conexiones únicas: {len(rutasValidas):,}")

aeropuertosRutas = pd.Index(
    pd.concat([rutasValidas["origen"], rutasValidas["destino"]], ignore_index=True).unique()
).sort_values()
mapaAeropuertosRutas = {
    idAeropuerto: indice
    for indice, idAeropuerto in enumerate(aeropuertosRutas)
}

indicesFilaRutas = rutasValidas["origen"].map(mapaAeropuertosRutas).to_numpy(dtype=np.int32)
indicesColumnaRutas = rutasValidas["destino"].map(mapaAeropuertosRutas).to_numpy(dtype=np.int32)
filasRutas = columnasRutas = len(aeropuertosRutas)

matrizRutasDispersa = csr_matrix(
    (
        np.ones(len(rutasValidas), dtype=np.float64),
        (indicesFilaRutas, indicesColumnaRutas)
    ),
    shape=(filasRutas, columnasRutas)
)

elementosNoNulosRutas = matrizRutasDispersa.nnz
totalPosicionesRutas = filasRutas * columnasRutas

densidadRutas = elementosNoNulosRutas / totalPosicionesRutas * 100

print("\nDatos de la matriz de rutas")
print(f"Aeropuertos (filas): {filasRutas:,}")
print(f"Aeropuertos (columnas): {columnasRutas:,}")
print(f"Dimensiones: {filasRutas} x {columnasRutas}")
print(f"Posiciones totales: {totalPosicionesRutas:,}")
print(f"Conexiones no nulas: {elementosNoNulosRutas:,}")
print(f"Densidad: {densidadRutas:.4f}%")
print(f"Porcentaje de ceros: {100 - densidadRutas:.4f}%")

matrizRutasDensa = matrizRutasDispersa.toarray()

bytesDensaRutas = matrizRutasDensa.nbytes
mbDensaRutas = bytesDensaRutas / (1024 * 1024)

print("\nMatriz densa")
print(f"Memoria: {mbDensaRutas:.4f} MB")
print(f"Memoria en bytes: {bytesDensaRutas:,}")

bytesDataRutas = matrizRutasDispersa.data.nbytes
bytesIndicesRutas = matrizRutasDispersa.indices.nbytes
bytesIndptrRutas = matrizRutasDispersa.indptr.nbytes

bytesCSRrutas = bytesDataRutas + bytesIndicesRutas + bytesIndptrRutas

mbCSRrutas = bytesCSRrutas / (1024 * 1024)

print("\nMatriz dispersa (CSR)")
print(f"Memoria de data: {bytesDataRutas:,} bytes")
print(f"Memoria de indices: {bytesIndicesRutas:,} bytes")
print(f"Memoria de indptr: {bytesIndptrRutas:,} bytes")
print(f"Memoria total: {mbCSRrutas:.4f} MB")
print(f"Memoria total en bytes: {bytesCSRrutas:,}")

ahorroRutas = (bytesDensaRutas - bytesCSRrutas) / bytesDensaRutas * 100

print("\nAhorro de memoria")
print(f"Ahorro de memoria con CSR: {ahorroRutas:.2f}%")

vectorRutas = np.ones(columnasRutas, dtype=np.float64)
resultadoRutas = matrizRutasDispersa.dot(vectorRutas)
resultadoRutasDenso = matrizRutasDensa.dot(vectorRutas)

print("\nMultiplicacion de matriz por vector")
print("Primeros 3 resultados:")
print(resultadoRutas[:3])

print(
    "Los resultados coinciden:",
    np.allclose(resultadoRutas, resultadoRutasDenso)
)
