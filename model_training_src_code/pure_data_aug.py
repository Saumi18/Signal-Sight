import numpy as np
import os
from full_data_extract import family_X, family_Y, len_family, family_map, mod_map

TOTAL_SAMPLES = 2**20
OUTPUT_DIR = 'processed_data'
OUTPUT_FILENAME = 'pure_only_raw_signals.npz'

family_names = list(family_map.keys())
X_signals = []
Y_labels = [] 

def normalize(sig):
    power_sig = sig.real if np.iscomplexobj(sig) else sig
    return sig / (np.sqrt(np.mean(power_sig**2)) + 1e-8)

print(f"Generating {TOTAL_SAMPLES} pure samples for the 24-class expert model...")
num_samples_per_class = TOTAL_SAMPLES // len(mod_map)

for mod_name, mod_family in mod_map.items():
    target_label_vec = family_Y[mod_family][0] 
    
    source_signals = []
    for i, label_vec in enumerate(family_Y[mod_family]):
        if np.array_equal(label_vec, target_label_vec):
            source_signals.append(family_X[mod_family][i])

    for _ in range(num_samples_per_class):
        idx = np.random.randint(len(source_signals))
        sig = source_signals[idx]
        
        X_signals.append(normalize(sig))
        Y_labels.append(np.argmax(target_label_vec))

# --- Save Raw Signal Data ---
X_signals = np.array(X_signals, dtype=np.float32)
Y_labels = np.array(Y_labels, dtype=np.int64)

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
print(f"\nSaving raw signals and integer labels to '{output_path}'...")
np.savez_compressed(output_path, X_signals=X_signals, Y_labels=Y_labels)
print(f"Shapes: X={X_signals.shape}, Y_labels={Y_labels.shape}")