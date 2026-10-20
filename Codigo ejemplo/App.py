import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from pathlib import Path

print("=========================================================")
print("Matriz dispersa de rutas aéreas desde CSV (OpenFlights)")
print("=========================================================")

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

rutaRutas = Path(__file__).resolve().parent / "routes.csv"

#* Buscamos el archivo routes.csv
try:
    datosRutas = pd.read_csv(
        rutaRutas,
        na_values=["\\N"]
    )

    #* organizamos los nombres de las columnas
    columnasNormalizadas = {
        str(col).strip().lower().replace("_", " ").replace("-", " "): col
        for col in datosRutas.columns
    }

    #* Buscamos las columnas
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
        (columnasNormalizadas[c] for c in opcionesOrigen
         if c in columnasNormalizadas),
        None
    )

    colDestino = next(
        (columnasNormalizadas[c] for c in opcionesDestino
         if c in columnasNormalizadas),
        None
    )

    #* Si el archivo no tiene nombre usamos los nombres de OpenFlights
    if colOrigen is None or colDestino is None:
        datosRutas = pd.read_csv(
            rutaRutas,
            header=None,
            names=nombresOpenFlights,
            na_values=["\\N"]
        )

        colOrigen = "source_airport_id"
        colDestino = "destination_airport_id"

except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo: {rutaRutas}")
    sys.exit(1)

except Exception as e:
    print(f"ERROR al cargar las rutas aéreas: {e}")
    sys.exit(1)


#* guardamos los codigos de los aeropuertos de cada ruta
rutasValidas = datosRutas[
    [colOrigen, colDestino]
].copy()

rutasValidas.columns = ["origen", "destino"]

#*  Convertimos los codigos en numeros
rutasValidas["origen"] = pd.to_numeric(
    rutasValidas["origen"], errors="coerce"
)

rutasValidas["destino"] = pd.to_numeric(
    rutasValidas["destino"], errors="coerce"
)

#* limpiamos las rutas que no tienen los dos codigos
rutasValidas = rutasValidas.dropna(
    subset=["origen", "destino"]
).copy()

#* Dejamos solo codigos enteros
rutasValidas = rutasValidas[
    (rutasValidas["origen"] % 1 == 0) &
    (rutasValidas["destino"] % 1 == 0)
].copy()

rutasValidas["origen"] = rutasValidas["origen"].astype(np.int64)
rutasValidas["destino"] = rutasValidas["destino"].astype(np.int64)

#* Borramos rutas que se repitan
rutasValidas = rutasValidas.drop_duplicates(
    subset=["origen", "destino"]
)

if rutasValidas.empty:
    print("ERROR: No se encontraron rutas aéreas válidas.")
    sys.exit(1)

print("\nListo, se cargaron las rutas.")
print(f"Conexiones únicas: {len(rutasValidas):,}")

#*  Juntamos todos los aeropuertos que aparecen en las rutas
aeropuertosRutas = pd.Index(
    pd.concat(
        [
            rutasValidas["origen"],
            rutasValidas["destino"]
        ],
        ignore_index=True
    ).unique()
).sort_values()

#* Le asignamos un numero a cada aeropuerto para usarlo en la matriz
mapaAeropuertosRutas = {
    idAeropuerto: indice
    for indice, idAeropuerto in enumerate(aeropuertosRutas)
}

indicesFilaRutas = rutasValidas["origen"].map(
    mapaAeropuertosRutas
).to_numpy(dtype=np.int32)

indicesColumnaRutas = rutasValidas["destino"].map(
    mapaAeropuertosRutas
).to_numpy(dtype=np.int32)

filasRutas = len(aeropuertosRutas)
columnasRutas = len(aeropuertosRutas)

#*   Ponemos un 1 donde hay una ruta
valoresRutas = np.ones(
    len(rutasValidas), dtype=np.float64
)

matrizRutasDispersa = csr_matrix(
    (
        valoresRutas,
        (indicesFilaRutas, indicesColumnaRutas)
    ),
    shape=(filasRutas, columnasRutas)
)

matrizRutasDispersa.eliminate_zeros()


elementosNoNulosRutas = matrizRutasDispersa.nnz
totalPosicionesRutas = filasRutas * columnasRutas

densidadRutas = (
    elementosNoNulosRutas / totalPosicionesRutas
) * 100

print("\nDatos de la matriz de rutas")
print(f"Aeropuertos (filas): {filasRutas:,}")
print(f"Aeropuertos (columnas): {columnasRutas:,}")
print(f"Dimensiones: {filasRutas} x {columnasRutas}")
print(f"Posiciones totales: {totalPosicionesRutas:,}")
print(f"Conexiones no nulas: {elementosNoNulosRutas:,}")
print(f"Densidad: {densidadRutas:.4f}%")
print(f"Porcentaje de ceros: {100 - densidadRutas:.4f}%")

#*  Hacemos una densa para comparar la memoria
matrizRutasDensa = matrizRutasDispersa.toarray()

bytesDensaRutas = matrizRutasDensa.nbytes
mbDensaRutas = bytesDensaRutas / (1024 * 1024)

print("\nMatriz densa")
print(f"Memoria: {mbDensaRutas:.4f} MB")
print(f"Memoria en bytes: {bytesDensaRutas:,}")

#*  Sumamos la memoria que usa la matriz CSR
bytesDataRutas = matrizRutasDispersa.data.nbytes
bytesIndicesRutas = matrizRutasDispersa.indices.nbytes
bytesIndptrRutas = matrizRutasDispersa.indptr.nbytes

bytesCSRrutas = (
    bytesDataRutas
    + bytesIndicesRutas
    + bytesIndptrRutas
)

mbCSRrutas = bytesCSRrutas / (1024 * 1024)

print("\nMatriz dispersa (CSR)")
print(f"Memoria de data: {bytesDataRutas:,} bytes")
print(f"Memoria de indices: {bytesIndicesRutas:,} bytes")
print(f"Memoria de indptr: {bytesIndptrRutas:,} bytes")
print(f"Memoria total CSR: {mbCSRrutas:.4f} MB")
print(f"Memoria total en bytes: {bytesCSRrutas:,}")

#*  Comparamos la memoria de las dos matrices
ahorroRutas = (
    (bytesDensaRutas - bytesCSRrutas)
    / bytesDensaRutas
) * 100

print("\nAhorro de memoria")
print(f"Ahorro de memoria con CSR: {ahorroRutas:.2f}%")

#*  Multiplicamos la matriz por un vector de unos
vectorRutas = np.ones(
    columnasRutas, dtype=np.float64
)

resultadoRutas = matrizRutasDispersa.dot(vectorRutas)
resultadoRutasDenso = matrizRutasDensa.dot(vectorRutas)

print("\nMultiplicacion de matriz por vector")
print("Primeros 3 resultados:")
print(resultadoRutas[:3])

print(
    "Los resultados coinciden:",
    np.allclose(resultadoRutas, resultadoRutasDenso)
)