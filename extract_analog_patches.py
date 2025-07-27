import numpy as np
import scipy.stats
import os

def spectral_entropy(patch, eps=1e-12):
    p = patch.flatten()
    p = p - np.min(p)
    p_sum = np.sum(p)
    if p_sum == 0:
        return 0
    p_norm = p / (p_sum + eps)
    return -np.sum(p_norm * np.log2(p_norm + eps))

def classify_analog_type(patch, entropy_threshold=4.5):
    ent = spectral_entropy(patch)
    return "FM" if ent > entropy_threshold else "AM"

def extract_analog_patch(spectrogram, patch_shape=(64, 32), lam_am=0.1, lam_fm=0.25, entropy_threshold=4.5):
    height, width = spectrogram.shape
    ph, pw = patch_shape
    max_score = -np.inf
    best_y, best_x = 0, 0
    for y in range(height - ph + 1):
        for x in range(width - pw + 1):
            patch = spectrogram[y:y+ph, x:x+pw]
            mean = np.mean(patch)
            var = np.var(patch)
            analog_type = classify_analog_type(patch, entropy_threshold)
            lam = lam_am if analog_type == "AM" else lam_fm
            score = mean - lam * var
            if score > max_score:
                max_score = score
                best_y, best_x = y, x
    return spectrogram[best_y:best_y+ph, best_x:best_x+pw], (best_y, best_x)

input_dir = '/home/taksh/Summer_RAID_Project/spectrograms/analog'
output_dir = '/home/taksh/Summer_RAID_Project/patches/analog_patches'
os.makedirs(output_dir, exist_ok=True)

patch_shape = (64, 32)
lam_am = 0.1        
lam_fm = 0.25      
entropy_threshold = 4.5  

spect_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])

for fname in spect_files:
    spect_path = os.path.join(input_dir, fname)
    Sxx = np.load(spect_path)
    patch, coords = extract_analog_patch(
        Sxx, patch_shape=patch_shape, lam_am=lam_am, lam_fm=lam_fm, entropy_threshold=entropy_threshold)
    patch_fname = fname.replace('spectrogram_', 'patch_')
    np.save(os.path.join(output_dir, patch_fname), patch)

print(f"Extracted {len(spect_files)} analog patches to {output_dir}")
