import os
import time
import glob
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import plotly.express as px

def log_tempo(inicio, msg):
    fim = time.time()
    print(f"⏱️ [{msg}] Tempo decorrido: {fim - inicio:.2f}s")
    return fim

def count_records(combined_file):
    inicio = time.time()
    print("\n📂 Carregando arquivo CSV...")
    df = pd.read_csv(combined_file, delimiter=";")
    inicio = log_tempo(inicio, "Leitura do arquivo CSV")
    df = df.dropna(subset=["cod_produtor_destino", "cod_produtor_origem"])
    df["cod_produtor_destino"] = df["cod_produtor_destino"].astype(str)
    df["cod_produtor_origem"]  = df["cod_produtor_origem"].astype(str)
    inicio = log_tempo(inicio, "Contagem de produtores")
    regs = df["cod_produtor_destino"].value_counts().reset_index()
    regs.columns = ["cod_produtor_destino", "count"]
    return regs, df

def preprocess_data(pid, df_all):
    inicio = time.time()
    f = df_all[df_all["cod_produtor_destino"] == pid].copy()
    f["dt_emissao_gta"] = pd.to_datetime(f["dt_emissao_gta"], errors='coerce')
    f = f.dropna(subset=["dt_emissao_gta"])
    f["data_seconds"] = f["dt_emissao_gta"].astype("int64") // 10**9
    f["repeticao_origem"] = f.groupby("cod_produtor_destino")["cod_produtor_origem"] \
                              .transform(lambda x: x.duplicated().astype(int))
    f["flag_binaria"] = np.where(f["repeticao_origem"] > 0, 1, 0)

    # ATENÇÃO: linhas sem distância_km devem ser removidas antes de escalar
    f = f.dropna(subset=["distancia_km"])

    scaler = MinMaxScaler()
    f[["qtd_norm","data_seconds_norm","distancia_km_norm"]] = scaler.fit_transform(
        f[["qtd","data_seconds","distancia_km"]]
    )
    log_tempo(inicio, "Pré-processamento")
    return f

def compute_distance_matrix(data):
    inicio = time.time()
    d = data.astype(np.float32)
    sq = np.sum(d**2, axis=1, keepdims=True)
    m = np.sqrt(np.maximum(0, sq - 2*np.dot(d, d.T) + sq.T))
    log_tempo(inicio, "Cálculo da matriz vetorizada")
    return m

def save_distance_matrix(matrix, pid):
    inicio = time.time()
    np.save(f"clusters/produtor_{pid}_dist_matrix.npy", matrix)
    log_tempo(inicio, "Salvamento da matriz")

def load_distance_matrix(pid):
    inicio = time.time()
    path = f"clusters/produtor_{pid}_dist_matrix.npy"
    if os.path.exists(path):
        m = np.load(path)
        log_tempo(inicio, "Carregamento da matriz")
        return m
    print("❌ Matriz não encontrada")
    return None

def cluster_producers(df, eps, min_samples, dist_matrix):
    inicio = time.time()
    if np.isnan(dist_matrix).any():
        raise ValueError("Há valores NaN na matriz de distâncias. Verifique o preprocessamento.")
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    df["cluster"] = db.fit_predict(dist_matrix)
    n = len(set(df["cluster"])) - (1 if -1 in df["cluster"] else 0)
    log_tempo(inicio, f"Clustering (eps={eps}, min={min_samples})")
    return df, n

def plot_clusters_tsne_3d(df, title):
    inicio = time.time()
    X = df[["qtd_norm","data_seconds_norm","distancia_km_norm"]].values
    if X.shape[0] < 3:
        print("Número insuficiente de amostras para t-SNE 3D.")
        return
    perp = min(30, X.shape[0]-1)
    tsne = TSNE(n_components=3, random_state=42, perplexity=perp, n_iter=1000)
    Y = tsne.fit_transform(X)
    df["tsne3d_1"], df["tsne3d_2"], df["tsne3d_3"] = Y.T

    fig = px.scatter_3d(
        df,
        x="tsne3d_1",
        y="tsne3d_2",
        z="tsne3d_3",
        color="cluster",
        labels={
            "tsne3d_1": "t-SNE 1",
            "tsne3d_2": "t-SNE 2",
            "tsne3d_3": "t-SNE 3"
        },
        title=title,
        opacity=0.7,
        height=700,
        width=900
    )
    fig.update_traces(marker=dict(size=4))
    fig.show()
    log_tempo(inicio, "Plotagem 3D interativa")

def salvar_outliers():
    arquivos = glob.glob("clusters/produtor_*_best.csv")
    frames = []
    for f in arquivos:
        df = pd.read_csv(f, sep=";")
        keep = [c for c in [
            "id_gta","dt_emissao_gta","cod_produtor_origem",
            "cod_produtor_destino","qtd","distancia_km","cluster"
        ] if c in df.columns]
        frames.append(df[keep])
    all_df = pd.concat(frames, ignore_index=True)
    out = all_df[all_df["cluster"] == -1]
    out.to_csv("clusters/possiveis_fraudes.csv", sep=";", index=False)
    print(f"🔍 Possíveis fraudes salvas ({len(out)})")

if __name__ == "__main__":
    inicio_total = time.time()

    combined_file = "/IMA/Deteccao_Fraude/bd/gtas_com_distancias.csv"
    eps_values = [0.3,0.5,0.7]
    min_samples_values = [2,3,5]
    regs, df_all = count_records(combined_file)
    os.makedirs("clusters", exist_ok=True)

    for i, pid in enumerate(regs["cod_produtor_destino"], 1):
        print(f"\n{'='*40}\n🚀 Produtor {i}/{len(regs)}: {pid}")
        proc = preprocess_data(pid, df_all)
        mat = load_distance_matrix(pid)
        if mat is None or mat.shape[0] != proc.shape[0]:
            mat = compute_distance_matrix(proc[["qtd_norm","data_seconds_norm","distancia_km_norm"]].values)
            save_distance_matrix(mat, pid)

        best = (-1, None, None, None)
        for eps in eps_values:
            for m in min_samples_values:
                clust, nc = cluster_producers(proc.copy(), eps, m, mat)
                if nc > best[0]:
                    best = (nc, eps, m, clust)

        if best[0] > -1:
            nc, eps, m, clust = best
            plot_clusters_tsne_3d(clust, f"Produtor {pid} – t-SNE 3D eps={eps}, m={m}")
            clust.to_csv(f"clusters/produtor_{pid}_best.csv", sep=";", index=False)
            print(f"✅ Configuração: eps={eps}, min={m} → {nc} clusters")

        log_tempo(inicio_total, f"Produtor {pid} completo")

    salvar_outliers()
    log_tempo(inicio_total, "Processamento total concluído")