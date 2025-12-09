import sqlite3
import pandas as pd
import numpy as np
import scipy.interpolate
import scipy.cluster.hierarchy as sch
import plotly.graph_objects as go
import os

DB_PATH = 'Materials/refractiveindex_sqlite/refractive.db'
OUTPUT_DIR = 'Material_Analysis'
MIN_WAVE_UM = 0.4
MAX_WAVE_UM = 0.7
POINTS = 100
BATCH_SIZE = 300

os.makedirs(OUTPUT_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("Fetching material list...")
cursor.execute("SELECT pageid, shelf, book, page FROM pages WHERE rangeMin <= ? AND rangeMax >= ?", (MIN_WAVE_UM, MAX_WAVE_UM))
materials = cursor.fetchall()
print(f"Found {len(materials)} materials.")

common_wave = np.linspace(MIN_WAVE_UM, MAX_WAVE_UM, POINTS)

data_n = {}
data_k = {}

print("Processing data...")
for pid, shelf, book, page in materials:
    name = f"{shelf}/{book}/{page}"
    
    cursor.execute("SELECT wave, refindex FROM refractiveindex WHERE pageid = ? ORDER BY wave", (pid,))
    rows_n = cursor.fetchall()
    if rows_n:
        w_n, n_val = zip(*rows_n)
        df_temp = pd.DataFrame({'w': w_n, 'n': n_val}).groupby('w').mean()
        if len(df_temp) > 1:
            f_n = scipy.interpolate.interp1d(df_temp.index, df_temp['n'], kind='linear', bounds_error=False, fill_value=np.nan)
            n_interp = f_n(common_wave)
            if not np.isnan(n_interp).any():
                 data_n[name] = n_interp

    cursor.execute("SELECT wave, coeff FROM extcoeff WHERE pageid = ? ORDER BY wave", (pid,))
    rows_k = cursor.fetchall()
    if rows_k:
        w_k, k_val = zip(*rows_k)
        df_temp = pd.DataFrame({'w': w_k, 'k': k_val}).groupby('w').mean()
        if len(df_temp) > 1:
            f_k = scipy.interpolate.interp1d(df_temp.index, df_temp['k'], kind='linear', bounds_error=False, fill_value=np.nan)
            k_interp = f_k(common_wave)
            if not np.isnan(k_interp).any():
                data_k[name] = k_interp

conn.close()

def process_and_plot(data_dict, quantity_name):
    if not data_dict:
        return

    df = pd.DataFrame(data_dict, index=common_wave)
    df = df.loc[:, df.std() > 1e-9]
    print(f"Processing {quantity_name}: {df.shape[1]} materials valid.")
    
    if df.shape[1] < 2:
        return

    print(f"Calculating correlation matrix for {quantity_name}...")
    corr = df.corr()
    
    print("Clustering...")
    try:
        d = sch.distance.pdist(df.T.values, metric='correlation')
        d = np.nan_to_num(d, nan=2.0)
        ind = sch.leaves_list(sch.linkage(d, method='average'))
        sorted_cols = [df.columns[i] for i in ind]
        corr = corr.loc[sorted_cols, sorted_cols]
    except Exception as e:
        print(f"Clustering failed: {e}")

    np.fill_diagonal(corr.values, np.nan)
    
    materials_list = corr.columns.tolist()
    
    for i in range(0, len(materials_list), BATCH_SIZE):
        batch_mats = materials_list[i : i + BATCH_SIZE]
        if not batch_mats:
            continue
            
        sub_corr = corr.loc[batch_mats, batch_mats]
        
        fig = go.Figure(data=go.Heatmap(
            z=sub_corr.values,
            x=batch_mats,
            y=batch_mats,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            hoverongaps=False
        ))
        
        axis_settings = dict(showticklabels=True) if len(batch_mats) <= 40 else dict(showticklabels=False)
        axis_settings['title'] = 'Materials'
        
        fig.update_layout(
            title=f'Correlation Matrix of {quantity_name} - Batch {i//BATCH_SIZE + 1}',
            xaxis=axis_settings,
            yaxis=axis_settings,
            width=1000, height=1000,
            template='plotly_white'
        )
        
        html_path = os.path.join(OUTPUT_DIR, f'heatmap_{quantity_name}_batch_{i//BATCH_SIZE + 1}.html')
        png_path = os.path.join(OUTPUT_DIR, f'heatmap_{quantity_name}_batch_{i//BATCH_SIZE + 1}.png')
        
        fig.write_html(html_path)
        try:
            fig.write_image(png_path)
            print(f"Saved {png_path} and {html_path}")
        except Exception as e:
             print(f"Saved {html_path} (PNG save failed: {e})")

process_and_plot(data_n, 'n')
process_and_plot(data_k, 'k')
