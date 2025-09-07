import numpy as np
from scipy.signal import spectrogram
from generate_spoofed_dataset import X_spoofed  # Import spoofed signals dictionary
import os


# Parameters (ensure consistent with clean and jammed spectrogram extraction)
fs = 1000
nperseg = 128
noverlap = 64


base_spec_dir = 'spectrograms/spoofed'
os.makedirs(base_spec_dir, exist_ok=True)


for family, signals in X_spoofed.items():
    family_dir = os.path.join(base_spec_dir, family)
    os.makedirs(family_dir, exist_ok=True)


    for idx, sig in enumerate(signals):
        # Handle IQ shape if needed (complex)
        if sig.ndim == 2 and sig.shape[1] == 2:
            sig = sig[:, 0] + 1j * sig[:, 1]
        
        f, t, Sxx = spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
        
        # Save spectrogram numpy array
        file_name = f'spectrogram_spoofed_{idx}.npy'
        np.save(os.path.join(family_dir, file_name), Sxx)


print(f"Spoofed spectrogram extraction complete. Files saved under '{base_spec_dir}/'")
