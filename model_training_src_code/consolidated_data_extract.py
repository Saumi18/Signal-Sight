import numpy as np
from scipy.signal import spectrogram
import os

from full_data_extract import family_X, family_Y, len_family, family_map, mod_map

TOTAL_SAMPLES = 2**18
PURE_RATIO = 0.5       # 50% pure, 50% mixed signals
NUM_PURE = int(TOTAL_SAMPLES * PURE_RATIO)
NUM_MIXED = TOTAL_SAMPLES - NUM_PURE

# --- Spectrogram Config ---
FS = 1000
NPERSEG = 64
NOVERLAP = 48
EPSILON = 1e-8

OUTPUT_DIR = 'processed_data'
ROUTER_SPECIALIST_FILENAME = 'spectrograms_and_labels.npz'
PURE_MIXED_FILENAME = 'pure_mixed_spectrograms.npz'
PURE_ONLY_FILENAME = 'pure_only_spectrograms.npz'

os.makedirs(OUTPUT_DIR, exist_ok=True)

family_names = list(family_map.keys())
mod_names = list(mod_map.keys())
num_families = len(family_names)



X_signals = []
Y_router = []      # For the Router model (multi-hot, len 3)
Y_special = []     # For the Specialist models (multi-hot, len 24)
Y_binary = []      # For the Pure-vs-Mixed model (0 or 1)
Y_pure_labels = [] # For the Pure-Only model (integer 0-23)

def normalize(sig):
    power_sig = sig.real if np.iscomplexobj(sig) else sig
    if sig.ndim == 2 and sig.shape[1] == 2: # Handle I/Q format
        power_sig = sig[:, 0]
    return sig / (np.sqrt(np.mean(power_sig**2)) + 1e-8)

print(f"Generating {NUM_PURE} pure samples...")
for _ in range(NUM_PURE):
    # Pick a random family, then a random signal from that family
    target_family = np.random.choice(family_names)
    idx = np.random.randint(len_family[target_family])
    sig = family_X[target_family][idx]
    special_label = family_Y[target_family][idx] # This is the original one-hot label (len 24)

    router_label = np.zeros(num_families, dtype=np.float32)
    router_label[family_names.index(target_family)] = 1.0

    X_signals.append(sig)
    Y_router.append(router_label)
    Y_special.append(special_label)
    Y_binary.append(0) # Label for pure is 0
    Y_pure_labels.append(np.argmax(special_label)) # Integer label 0-23

print(f"Generating {NUM_MIXED} mixed samples...")
for _ in range(NUM_MIXED):
    fam1_name, fam2_name = np.random.choice(family_names, size=2, replace=False)

    idx1, idx2 = np.random.randint(len_family[fam1_name]), np.random.randint(len_family[fam2_name])
    sig1, label1 = family_X[fam1_name][idx1], family_Y[fam1_name][idx1]
    sig2, label2 = family_X[fam2_name][idx2], family_Y[fam2_name][idx2]

    power_ratio_db = np.random.uniform(-3, 3)
    scale_factor = 10**(power_ratio_db / 20.0)
    mixed_sig = normalize(normalize(sig1) + scale_factor * normalize(sig2))

    mixed_special_label = label1 + label2
    mixed_router_label = np.zeros(num_families, dtype=np.float32)
    mixed_router_label[family_names.index(fam1_name)] = 1.0
    mixed_router_label[family_names.index(fam2_name)] = 1.0

    X_signals.append(mixed_sig)
    Y_router.append(mixed_router_label)
    Y_special.append(mixed_special_label)
    Y_binary.append(1) # Label for mixed is 1
    Y_pure_labels.append(-1) # Placeholder, as this sample is not pure

Y_router = np.array(Y_router, dtype=np.float32)
Y_special = np.array(Y_special, dtype=np.float32)
Y_binary = np.array(Y_binary, dtype=np.int64)
Y_pure_labels = np.array(Y_pure_labels, dtype=np.int64)


global_min, global_max = float("inf"), float("-inf")
for i, sig in enumerate(X_signals):
    # Recombine I/Q data into a complex signal if necessary
    if sig.ndim == 2 and sig.shape[1] == 2:
        sig = sig[:, 0] + 1j * sig[:, 1]
        
    _, _, Sxx_complex = spectrogram(sig, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    Sxx_log = np.log1p(np.abs(Sxx_complex))
    
    global_min, global_max = min(global_min, Sxx_log.min()), max(global_max, Sxx_log.max())
    if (i + 1) % 20000 == 0: print(f"  ...scanned {i+1}/{len(X_signals)} signals")

print(f"Universal Normalization Constants Found: MIN={global_min:.8f}, MAX={global_max:.8f}")


all_spectrograms = []
for i, sig in enumerate(X_signals):
    # Recombine I/Q data into a complex signal if necessary
    if sig.ndim == 2 and sig.shape[1] == 2:
        sig = sig[:, 0] + 1j * sig[:, 1]

    _, _, Sxx_complex = spectrogram(sig, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    Sxx_log = np.log1p(np.abs(Sxx_complex))
    
    # Normalize with the hardcoded universal constants
    Sxx_norm = (Sxx_log - global_min) / (global_max - global_min + EPSILON)
    
    all_spectrograms.append(Sxx_norm)
    if (i + 1) % 20000 == 0: print(f"  ...generated {i+1}/{len(X_signals)} spectrograms")

X_spectrograms = np.array(all_spectrograms, dtype=np.float32)


# --- Dataset 1: For Router and Specialists ---
path = os.path.join(OUTPUT_DIR, ROUTER_SPECIALIST_FILENAME)
print(f"Saving Router/Specialist data to '{path}'...")
np.savez_compressed(
    path,
    X_spectrograms=X_spectrograms,
    Y_router=Y_router,
    Y_special=Y_special
)

# --- Dataset 2: For Pure-vs-Mixed Classifier ---
path = os.path.join(OUTPUT_DIR, PURE_MIXED_FILENAME)
print(f"Saving Pure-vs-Mixed data to '{path}'...")
np.savez_compressed(
    path,
    X_spectrograms=X_spectrograms,
    Y_binary=Y_binary
)

# --- Dataset 3: For Pure-Only CNN ---
path = os.path.join(OUTPUT_DIR, PURE_ONLY_FILENAME)
print(f"Saving Pure-Only data to '{path}'...")
# Filter to include only the pure signals
pure_mask = (Y_binary == 0)
X_pure = X_spectrograms[pure_mask]
Y_pure = Y_pure_labels[pure_mask]
np.savez_compressed(
    path,
    X_spectrograms=X_pure,
    Y_labels=Y_pure
)

print("\n--- All processing complete. ---")