import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.cluster.hierarchy as sch
import os

DB_PATH = 'Materials/refractiveindex_sqlite/refractive.db'
OUTPUT_DIR = 'Visualization/Outputs'
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

print("Fetching and interpolating data...")
for pid, shelf, book, page in materials:
    name = f"{shelf}/{book}/{page}"
    
    cursor.execute("SELECT wave, refindex FROM refractiveindex WHERE pageid = ? ORDER BY wave", (pid,))
    rows_n = cursor.fetchall()
    if rows_n:
        w_n, n_val = zip(*rows_n)
        df_temp = pd.DataFrame({'w': w_n, 'n': n_val}).groupby('w').mean()
        w_n_u = df_temp.index.values
        n_val_u = df_temp['n'].values
        
        if len(w_n_u) > 1:
            f_n = scipy.interpolate.interp1d(w_n_u, n_val_u, kind='linear', bounds_error=False, fill_value=np.nan)
            n_interp = f_n(common_wave)
            if not np.isnan(n_interp).any():
                 data_n[name] = n_interp

    cursor.execute("SELECT wave, coeff FROM extcoeff WHERE pageid = ? ORDER BY wave", (pid,))
    rows_k = cursor.fetchall()
    if rows_k:
        w_k, k_val = zip(*rows_k)
        df_temp = pd.DataFrame({'w': w_k, 'k': k_val}).groupby('w').mean()
        w_k_u = df_temp.index.values
        k_val_u = df_temp['k'].values
        
        if len(w_k_u) > 1:
            f_k = scipy.interpolate.interp1d(w_k_u, k_val_u, kind='linear', bounds_error=False, fill_value=np.nan)
            k_interp = f_k(common_wave)
            if not np.isnan(k_interp).any():
                data_k[name] = k_interp

conn.close()

def process_and_plot(data_dict, quantity_name):
    if not data_dict:
        print(f"No data for {quantity_name}")
        return

    df = pd.DataFrame(data_dict, index=common_wave)
    df = df.loc[:, df.std() > 1e-9]
    print(f"Processing {quantity_name}: {df.shape[1]} materials valid.")
    
    if df.shape[1] < 2:
        print("Not enough materials for correlation.")
        return

    print(f"Calculating correlation matrix for {quantity_name}...")
    corr = df.corr()
    
    print("Clustering to determine sort order...")
    try:
        d = sch.distance.pdist(df.T.values, metric='correlation')
        d = np.nan_to_num(d, nan=2.0)
        L = sch.linkage(d, method='average')
        ind = sch.leaves_list(L)
        
        sorted_cols = [df.columns[i] for i in ind]
        corr = corr.loc[sorted_cols, sorted_cols]
        print("Correlation matrix sorted.")
    except Exception as e:
        print(f"Clustering failed: {e}. Keeping original order.")
        sorted_cols = df.columns.tolist()

    np.fill_diagonal(corr.values, np.nan)
    
    materials_list = corr.columns.tolist()
    num_materials = len(materials_list)
    
    for i in range(0, num_materials, BATCH_SIZE):
        batch_mats = materials_list[i : i + BATCH_SIZE]
        if not batch_mats:
            continue
            
        sub_corr = corr.loc[batch_mats, batch_mats]
        
        plt.figure(figsize=(12, 10))
        current_data = sub_corr.values
        
        plt.imshow(current_data, cmap='coolwarm', aspect='auto', interpolation='nearest', vmin=-1, vmax=1)
        plt.colorbar(label=f'Correlation of {quantity_name}')
        plt.title(f'Correlation Matrix of {quantity_name} - Batch {i//BATCH_SIZE + 1}')
        
        if len(batch_mats) <= 30:
            plt.xticks(range(len(batch_mats)), batch_mats, rotation=90, fontsize=8)
            plt.yticks(range(len(batch_mats)), batch_mats, fontsize=8)
        else:
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(f"Materials (Sorted Batch {i//BATCH_SIZE + 1})")
            plt.ylabel(f"Materials (Sorted Batch {i//BATCH_SIZE + 1})")

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f'heatmap_{quantity_name}_batch_{i//BATCH_SIZE + 1}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")

process_and_plot(data_n, 'n')
process_and_plot(data_k, 'k')