import numpy as np
import scipy.stats
import os

def extract_qam_patch(spectrogram, patch_shape=(64, 32), lam=0.2):
    height, width = spectrogram.shape
    ph, pw = patch_shape
    max_score = -np.inf
    best_y, best_x = 0, 0
    for y in range(height - ph +1):
        for x in range(width - pw + 1):
            patch=spectrogram[y:y:y+ph, x:x+pw].flatten()
            var=np.var(patch)
            kurt=scipy.stats.kurtosis(patch)
            score = var - lam * np.abs(kurt)
            if score > max_score:
                max_score = score
                best_y, best_x = y, x
    return spectrogram[best_y:best_y+ph, best_x:best_x+pw], (best_y,best_x)

input_dir = '/home/taksh/Summer_RAID_Project/spectrograms/qam'
output_dir = '/home/taksh/Summer_RAID_Project/patches/qam_patches'
os.makedirs(output_dir, exist_ok=True)

patch_shape = (64, 32)
lam = 0.2

spect_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])

for fname in spect_files:
    spect_path = os.path.join(input_dir, fname)
    Sxx = np.load(spect_path)
    patch, coords = extract_qam_patch(Sxx, patch_shape=patch_shape, lam=lam)
    patch_fname = fname.replace('spectrogram_', 'patch_')
    np.save(os.path.join(output_dir, patch_fname), patch)

print(f"Extracted {len(spect_files)} QAM patches to {output_dir}")