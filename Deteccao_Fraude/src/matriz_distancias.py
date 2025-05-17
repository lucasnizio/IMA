import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

DRIVE_HOST   = os.getenv("OSRM_DRIVE_HOST", "http://localhost:5000")
WALK_HOST    = os.getenv("OSRM_WALK_HOST",  "http://localhost:5001")
MUNIS_CSV    = "/IMA/Deteccao_Fraude/bd/municipios.csv"
OUTPUT_CSV   = "/IMA/Deteccao_Fraude/bd/matriz_distancias.csv"
MAX_WORKERS  = os.cpu_count() or 6

def load_municipios(path):
    m = pd.read_csv(path)
    m.columns = m.columns.str.strip().str.lower()
    m = m.rename(columns={
        'nome': 'municipio',
        'latitude': 'lat',
        'longitude': 'lon'
    })
    return m[['municipio','lat','lon','codigo_uf']]

def build_pairs(m):
    mg   = m[m.codigo_uf == 31][['municipio','lat','lon']]
    all_ = m[['municipio','lat','lon']].copy()
    mg   = mg.rename(columns={
        'municipio':'orig_mun',
        'lat':'orig_lat',
        'lon':'orig_lon'
    })
    all_ = all_.rename(columns={
        'municipio':'dest_mun',
        'lat':'dest_lat',
        'lon':'dest_lon'
    })
    return mg.merge(all_, how='cross')

def fetch(profile, host, o_lon, o_lat, d_lon, d_lat):
    url = f"{host}/route/v1/{profile}/{o_lon},{o_lat};{d_lon},{d_lat}?overview=false"
    r = requests.get(url, timeout=10).json()
    if r.get("code") == "Ok":
        rt = r['routes'][0]
        return rt['distance'] / 1000.0, rt['duration'] / 60.0
    return None, None

def worker(idx, total, row):
    o_lon, o_lat = row.orig_lon, row.orig_lat
    d_lon, d_lat = row.dest_lon, row.dest_lat
    dr_km, dr_min = fetch('driving', DRIVE_HOST, o_lon, o_lat, d_lon, d_lat)
    wk_km, wk_min = fetch('foot',    WALK_HOST,  o_lon, o_lat, d_lon, d_lat)

    return {
        'municipio_origem':  row.orig_mun,
        'municipio_destino': row.dest_mun,
        'dist_rod_km':       dr_km,
        'dist_rod_min':      dr_min,
        'dist_ped_km':       wk_km,
        'dist_ped_min':      wk_min
    }

def main():
    mus  = load_municipios(MUNIS_CSV)
    pairs = build_pairs(mus)
    total = len(pairs)
    print(f"→ Total de pares: {total}")

    pd.DataFrame(columns=[
        'municipio_origem','municipio_destino',
        'dist_rod_km','dist_rod_min',
        'dist_ped_km','dist_ped_min'
    ]).to_csv(OUTPUT_CSV, sep=';', index=False)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        tasks = (
            (i+1, total, row)
            for i, row in enumerate(pairs.itertuples(index=False))
        )
        futures = { pool.submit(worker, *t): None for t in tasks }

        for fut in as_completed(futures):
            out = fut.result()
            pd.DataFrame([out]).to_csv(
                OUTPUT_CSV, sep=';', mode='a',
                index=False, header=False
            )

    print(f"✓ Dados em {OUTPUT_CSV}")

if __name__ == "__main__":
    main()