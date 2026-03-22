import numpy as np
import json
import os
import matplotlib.pyplot as plt

data_dir = r"./data/sub-03/ses-02/run-04"

def check_npy(filename):
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"[MISSING] {filename}")
        return

    try:
        data = np.load(path)
        print(f"\n--- {filename} ---")
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        
        if np.issubdtype(data.dtype, np.number):
            print(f"Min: {np.nanmin(data)}, Max: {np.nanmax(data)}")
            print(f"Has NaNs: {np.isnan(data).any()}")
            print(f"Has Infs: {np.isinf(data).any()}")
        else:
            print(f"Sample: {data[:5]}")
            
    except Exception as e:
        print(f"[ERROR] Could not load {filename}: {e}")

def check_json(filename):
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"[MISSING] {filename}")
        return

    try:
        with open(path, 'r') as f:
            data = json.load(f)
        print(f"\n--- {filename} ---")
        print(json.dumps(data, indent=2)[:500] + "..." if len(json.dumps(data)) > 500 else json.dumps(data, indent=2))
    except Exception as e:
        print(f"[ERROR] Could not load {filename}: {e}")

print(f"Checking data in: {data_dir}")
check_npy("eeg.npy")
check_npy("aux.npy")
check_npy("timestamp.npy")
check_json("markers.json")
check_json("metadata.json")

run_folder_path = data_dir

aux_path = run_folder_path + 'aux.npy'
if not os.path.exists(aux_path):
    print('No aux.npy for this run')
else:
    aux = np.load(aux_path)
    if aux.size == 0:
        print('aux.npy exists but is empty')
    else:
        # run_nnt_experiment stores shape (channels, samples)
        if aux.shape[0] <= 64 and aux.shape[1] > aux.shape[0]:
            aux = aux.T  # make (samples, channels) for plotting

        print('aux shape (samples,channels):', aux.shape)
        plt.figure(figsize=(12, 4))
        for ch in range(aux.shape[1]):
            plt.plot(aux[:, ch] + ch * 1.2 * np.nanmax(np.abs(aux)), label=f'aux{ch}')
        for m in markers:
            plt.axvline(m['start_sample_index'], color='red', alpha=0.3)
        plt.legend(); plt.show()
