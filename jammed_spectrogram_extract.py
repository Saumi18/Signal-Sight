import numpy as np
from scipy.signal import spectrogram
from generate_jammed_dataset import X_jammed  # Import jammed signals dictionary
import os

# Parameters (make sure these match your clean spectrogram parameters)
fs = 1000
nperseg = 128
noverlap = 64

base_spec_dir = 'spectrograms/jammed'
os.makedirs(base_spec_dir, exist_ok=True)

for family, signals in X_jammed.items():
    family_dir = os.path.join(base_spec_dir, family)
    os.makedirs(family_dir, exist_ok=True)

    for idx, sig in enumerate(signals):
        # Handle IQ shape if needed (complex)
        if sig.ndim == 2 and sig.shape[1] == 2:
            sig = sig[:, 0] + 1j * sig[:, 1]
        
        f, t, Sxx = spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
        
        # Save spectrogram numpy array
        file_name = f'spectrogram_jammed_{idx}.npy'
        np.save(os.path.join(family_dir, file_name), Sxx)

print(f"Jammed spectrogram extraction complete. Files saved under '{base_spec_dir}/'")
