import pandas as pd

def pivot_faixas(input_csv, tipo):
    df = pd.read_csv(input_csv, sep=';', encoding='utf-8')
    if tipo == 'oe':
        df = df.rename(columns={'id_gta_origem_outro_estado': 'id_gta'})
    pivot = (
        df
        .pivot_table(
            index='id_gta',
            columns='ds_faixa_etaria',
            values='qt_animais',
            aggfunc='sum',
            fill_value=0
        )
        .reset_index()
    )
    return pivot

def validar_soma_faixas(faixa, gtas_csv, tipo):
    gtas = pd.read_csv(gtas_csv, sep=';', encoding='utf-8')
    if tipo == 'oe':
        gtas = gtas.rename(columns={'id_gta_origem_outro_estado': 'id_gta'})
    colunas_faixas = [
        'Acima de 36 meses',
        'De 0 até 12 meses',
        'De 13 até 24 meses',
        'De 25 até 36 meses'
    ]
    faixa['soma_faixas'] = faixa[colunas_faixas].sum(axis=1)
    merged = pd.merge(gtas, faixa, on='id_gta', how='left')
    merged.drop(columns=['soma_faixas'], inplace=True)
    return merged

def combinar_gtas(gtas_oe, gtas_mg):
    gtas_oe.rename(columns={'municipio_origem': 'nome_municipio_origem'}, inplace=True)
    gtas_oe.rename(columns={'municipio_destino': 'nome_municipio_destino'}, inplace=True)
    merged = pd.concat([gtas_mg, gtas_oe], ignore_index=True)
    return merged

def combinar_distancias_gtas(dist_csv, df_gta):
    df_dist = pd.read_csv(dist_csv, sep=';', encoding='utf-8')
    df_dist['municipio_origem']   = df_dist['municipio_origem'].str.upper()
    df_dist['municipio_destino']  = df_dist['municipio_destino'].str.upper()
    df = df_gta.merge(df_dist,on=['nome_municipio_origem','nome_municipio_destino'],how='left')
    df['distancia_km'] = df.apply(lambda r: r['dist_ped_km'] if r['ds_meio_transporte']=='A PÉ' else r['dist_rod_km'],axis=1)
    df['duracao_min'] = df.apply(lambda r: r['dist_ped_min'] if r['ds_meio_transporte']=='A PÉ'else r['dist_rod_min'],axis=1)
    df = df.drop(columns=['dist_rod_km','dist_rod_min','dist_ped_km','dist_ped_min'])
    df.to_csv('/IMA/Deteccao_Fraude/bd/gtas_com_distancias.csv', sep=';', index=False, encoding='utf-8')

if __name__ == '__main__':
    gtas_oe = validar_soma_faixas(pivot_faixas('/IMA/Deteccao_Fraude/bd/bd_gta_oe_faixa_etaria202505091630.csv', 'oe'), '/IMA/Deteccao_Fraude/bd/bd_gta_oe202505091618.csv', 'oe')
    gtas_mg = validar_soma_faixas(pivot_faixas('/IMA/Deteccao_Fraude/bd/bd_gta_dentro_mg_faixa_etaria202505091602.csv', ''), '/IMA/Deteccao_Fraude/bd/bd_gta_dentro_mg202505091607.csv', '')
    gtas_combinada = combinar_gtas(gtas_oe, gtas_mg)
    combinar_distancias_gtas('/IMA/Deteccao_Fraude/bd/matriz_distancias.csv', gtas_combinada)
