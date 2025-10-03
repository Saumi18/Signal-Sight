import numpy as np
import os
from full_data_extract import family_X, family_Y, len_family, family_map

TOTAL_SAMPLES = 2**18  # 131,072 samples should be sufficient for this binary task
PURE_RATIO = 0.5       # Create a balanced 50/50 dataset
NUM_PURE = int(TOTAL_SAMPLES * PURE_RATIO)
NUM_MIXED = TOTAL_SAMPLES - NUM_PURE
OUTPUT_DIR = 'processed_data'
OUTPUT_FILENAME = 'pure_mixed_raw_signals.npz'

family_names = list(family_map.keys())
num_families = len(family_names)

X_signals = []
Y_binary = []  # Label: 0 for Pure, 1 for Mixed

def normalize(sig):
    power_sig = sig.real if np.iscomplexobj(sig) else sig
    return sig / (np.sqrt(np.mean(power_sig**2)) + 1e-8)

print(f"Generating {NUM_PURE} pure samples...")
for _ in range(NUM_PURE):
    target_family = np.random.choice(family_names)
    idx = np.random.randint(len_family[target_family])
    sig = family_X[target_family][idx]
    
    X_signals.append(normalize(sig))
    Y_binary.append(0)

print(f"Generating {NUM_MIXED} mixed samples...")
for _ in range(NUM_MIXED):
    fam1_name, fam2_name = np.random.choice(family_names, size=2, replace=False)
    idx1 = np.random.randint(len_family[fam1_name])
    idx2 = np.random.randint(len_family[fam2_name])
    sig1 = normalize(family_X[fam1_name][idx1])
    sig2 = normalize(family_X[fam2_name][idx2])
    power_ratio_db = np.random.uniform(-3, 3)
    scale_factor = 10**(power_ratio_db / 20.0)
    mixed_sig = normalize(sig1 + scale_factor * sig2)
    
    X_signals.append(mixed_sig)
    Y_binary.append(1)

X_signals = np.array(X_signals, dtype=np.float32)
Y_binary = np.array(Y_binary, dtype=np.int64)

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
print(f"\nSaving raw signals and binary labels to '{output_path}'...")
np.savez_compressed(output_path, X_signals=X_signals, Y_binary=Y_binary)
print(f"Shapes: X={X_signals.shape}, Y_binary={Y_binary.shape}")