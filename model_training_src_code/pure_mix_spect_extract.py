import numpy as np
from scipy.signal import spectrogram
import os

INPUT_DIR = 'processed_data'
INPUT_FILENAME = 'pure_mixed_raw_signals.npz'
OUTPUT_FILENAME = 'pure_mixed_spectrograms.npz'
FS, NPERSEG, NOVERLAP, EPSILON = 1000, 64, 48, 1e-8

# Load the raw signal data
print(f"Loading raw data from '{os.path.join(INPUT_DIR, INPUT_FILENAME)}'...")
data = np.load(os.path.join(INPUT_DIR, INPUT_FILENAME))
X, Y_binary = data['X_signals'], data['Y_binary']

# Pass 1: Compute global min/max
print("Pass 1: Scanning for global min/max...")
global_min, global_max = float("inf"), float("-inf")
for sig in X:
    _, _, Sxx = spectrogram(sig, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    Sxx_log = np.log1p(np.abs(Sxx))
    global_min, global_max = min(global_min, Sxx_log.min()), max(global_max, Sxx_log.max())
print(f"Global min={global_min:.4f}, max={global_max:.4f}")

# Pass 2: Normalize and collect
print("Pass 2: Generating normalized spectrograms...")
all_spectrograms = []
for sig in X:
    _, _, Sxx = spectrogram(sig, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    Sxx_log = np.log1p(np.abs(Sxx))
    Sxx_norm = (Sxx_log - global_min) / (global_max - global_min + EPSILON)
    all_spectrograms.append(Sxx_norm)

X_spectrograms = np.array(all_spectrograms, dtype=np.float32)

# Pass 3: Save final data
output_path = os.path.join(INPUT_DIR, OUTPUT_FILENAME)
print(f"Saving spectrograms to '{output_path}'...")
np.savez_compressed(output_path, X_spectrograms=X_spectrograms, Y_binary=Y_binary)
print("\nProcessing complete.")
print(f"Final shapes: X={X_spectrograms.shape}, Y={Y_binary.shape}")