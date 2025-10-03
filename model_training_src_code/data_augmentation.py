import numpy as np
from full_data_extract import family_X, family_Y, len_family, family_map

family_names = list(family_map.keys())
num_families = len(family_names)

TOTAL_SAMPLES = 2**18  
PURE_RATIO = 0.4       # 40% of samples will be pure signals
NUM_PURE = int(TOTAL_SAMPLES * PURE_RATIO)
NUM_MIXED = TOTAL_SAMPLES - NUM_PURE

X = []
Y_special = [] # Will store detailed multi-hot labels (length 24)
Y_router = []  # Will store family-level multi-hot labels (length 3)

def normalize(sig):
    """ Normalizes the signal power. """
    # Ensure the signal is real before calculating power for normalization
    power_sig = sig.real if np.iscomplexobj(sig) else sig
    return sig / (np.sqrt(np.mean(power_sig**2)) + 1e-8)

# --- 1. Generate Pure Samples ---
print(f"Generating {NUM_PURE} pure samples...")
for target_family in family_names:
    num_samples_for_family = NUM_PURE // num_families
    for _ in range(num_samples_for_family):
        idx = np.random.randint(len_family[target_family])
        sig = family_X[target_family][idx]
        
        special_label = family_Y[target_family][idx]

        router_label = np.zeros(num_families, dtype=np.float32)
        router_label[family_names.index(target_family)] = 1.0

        X.append(normalize(sig))
        Y_special.append(special_label)
        Y_router.append(router_label)

print(f"Generating {NUM_MIXED} mixed samples...")
for _ in range(NUM_MIXED):
    fam1_name, fam2_name = np.random.choice(family_names, size=2, replace=False)

    idx1 = np.random.randint(len_family[fam1_name])
    idx2 = np.random.randint(len_family[fam2_name])
    
    sig1 = normalize(family_X[fam1_name][idx1])
    label1 = family_Y[fam1_name][idx1] # One-hot label for sig1
    
    sig2 = normalize(family_X[fam2_name][idx2])
    label2 = family_Y[fam2_name][idx2] # One-hot label for sig2

    # Mix signals with a random power ratio between -3 dB and +3 dB
    power_ratio_db = np.random.uniform(-3, 3)
    scale_factor = 10**(power_ratio_db / 20.0)
    mixed_sig = normalize(sig1 + scale_factor * sig2)

    mixed_special_label = label1 + label2

    mixed_router_label = np.zeros(num_families, dtype=np.float32)
    mixed_router_label[family_names.index(fam1_name)] = 1.0
    mixed_router_label[family_names.index(fam2_name)] = 1.0

    X.append(mixed_sig)
    Y_special.append(mixed_special_label)
    Y_router.append(mixed_router_label)


X = np.array(X, dtype=np.float32)
Y_special = np.array(Y_special, dtype=np.float32)
Y_router = np.array(Y_router, dtype=np.float32)

print("\nData augmentation complete.")
print(f"Shapes: X={X.shape}, Y_special={Y_special.shape}, Y_router={Y_router.shape}")
