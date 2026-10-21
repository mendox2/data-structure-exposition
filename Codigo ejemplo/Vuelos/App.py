import sys
import pandas as pd


try:
    rutas = pd.read_csv("Codigo ejemplo/routes.csv", na_values=["\\N"])
except FileNotFoundError:
    print("ERROR: No se encontró Codigo ejemplo/routes.csv.")
    sys.exit(1)
except Exception as error:
    print(f"ERROR al cargar las rutas: {error}")
    sys.exit(1)

rutas = rutas.dropna(subset=["source airport", "destination airport"])
rutas["source airport"] = rutas["source airport"].str.upper()
rutas["destination airport"] = rutas["destination airport"].str.upper()

vuelos = {}
for origen, destino in rutas[
    ["source airport", "destination airport"]
].itertuples(index=False, name=None):
    vuelos.setdefault(origen, []).append(destino)

codigosIATA = sorted(set(rutas["source airport"]) | set(rutas["destination airport"]))
print("Códigos IATA disponibles:")
print(", ".join(codigosIATA))

origen = input("\nAeropuerto de origen: ").strip().upper()
destino = input("Aeropuerto de destino: ").strip().upper()

if origen not in codigosIATA or destino not in codigosIATA:
    print("ERROR: Usa códigos IATA de la lista.")
elif origen == destino:
    print("El origen y el destino son el mismo aeropuerto.")
else:
    pendientes = [origen]
    anterior = {origen: None}
    indice = 0

    while indice < len(pendientes) and destino not in anterior:
        aeropuerto = pendientes[indice]
        indice += 1

        for siguiente in vuelos.get(aeropuerto, []):
            if siguiente not in anterior:
                anterior[siguiente] = aeropuerto
                pendientes.append(siguiente)

    if destino not in anterior:
        print(f"No se encontró una ruta de {origen} a {destino}.")
    else:
        ruta = []
        aeropuerto = destino
        while aeropuerto is not None:
            ruta.append(aeropuerto)
            aeropuerto = anterior[aeropuerto]
        ruta.reverse()

        print(f"La ruta con menos viajes es: {ruta}")
        print(f"Vuelos: {len(ruta) - 1}")
