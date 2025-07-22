import numpy as np
from scipy.signal import spectrogram
from generate_input_data import X
import os

os.makedirs('spectrograms', exist_ok=True)

#parameters
fs = 1000
nperseg = 128 
noverlap = 64

for family, signals in X.items():
    family_dir = os.path.join('spectrograms', family)
    os.makedirs(family_dir, exist_ok=True)

    for idx, sig in enumerate(signals):
            if sig.ndim == 2 and sig.shape[1] == 2:
                sig = sig[:, 0] + 1j * sig[:, 1]
            f, t, Sxx = spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
            np.save(os.path.join(family_dir, f'spectrogram_{idx}.npy'), Sxx)
print('Spectrogram extraction complete. Spectrograms saved in spectrograms/ directory.')



