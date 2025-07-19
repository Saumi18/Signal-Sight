import numpy as np
from scipy.signal import spectrogram
from full_data_extract import family_X, len_family
import os

# Directory to save spectrogram arrays
os.makedirs('spectrograms', exist_ok=True)

# Parameters for spectrogram (adjust as needed)
fs = 1000  # Example sampling frequency
nperseg = 128
noverlap = 64

for family, signals in family_X.items():
    family_dir = os.path.join('spectrograms', family)
    os.makedirs(family_dir, exist_ok=True)
    for idx, sig in enumerate(signals):
        f, t, Sxx = spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
        # Save the spectrogram as a numpy array
        np.save(os.path.join(family_dir, f'spectrogram_{idx}.npy'), Sxx)

print('Spectrogram extraction complete. Spectrograms saved in spectrograms/ directory.')
