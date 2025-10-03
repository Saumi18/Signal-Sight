import numpy as np
from scipy.signal import spectrogram
import os

from data_augmentation import X, Y_router, Y_special

OUTPUT_DIR = 'processed_data'
OUTPUT_FILENAME = 'spectrograms_and_labels.npz'
FS = 1000  # Sample rate
NPERSEG = 64  # Length of each segment for STFT
NOVERLAP = 48 # Overlap between segments
EPSILON = 1e-8 # To prevent division by zero

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Data will be saved in '{OUTPUT_DIR}'...")

# -------------------------------------------------------------------
# Pass 1: Compute global min/max for consistent normalization
# -------------------------------------------------------------------
print("Pass 1: Scanning dataset for global min/max for normalization...")
global_min, global_max = float("inf"), float("-inf")
for sig in X:
    # Ensure signal is complex for spectrogram calculation
    if sig.ndim == 2 and sig.shape[1] == 2:
        sig = sig[:, 0] + 1j * sig[:, 1]
        
    _, _, Sxx_complex = spectrogram(sig, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    Sxx_log = np.log1p(np.abs(Sxx_complex)) # Log-magnitude
    
    global_min = min(global_min, Sxx_log.min())
    global_max = max(global_max, Sxx_log.max())

print(f"Global min={global_min:.4f}, max={global_max:.4f}")

# -------------------------------------------------------------------
# Pass 2: Normalize and collect all spectrograms
# -------------------------------------------------------------------
print("Pass 2: Normalizing and generating all spectrograms...")
all_spectrograms = []
for idx, sig in enumerate(X):
    # --- Generate and Normalize Spectrogram ---
    if sig.ndim == 2 and sig.shape[1] == 2:
        sig = sig[:, 0] + 1j * sig[:, 1]

    _, _, Sxx_complex = spectrogram(sig, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    Sxx_log = np.log1p(np.abs(Sxx_complex))
    Sxx_norm = (Sxx_log - global_min) / (global_max - global_min + EPSILON)
    
    all_spectrograms.append(Sxx_norm)

# Convert the list of spectrograms to a single NumPy array
X_spectrograms = np.array(all_spectrograms, dtype=np.float32)

# -------------------------------------------------------------------
# Pass 3: Save all data into a single compressed file
# -------------------------------------------------------------------
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
print(f"Saving all data to '{output_path}'...")

# Using .npz to save multiple arrays in one file. It's efficient.
np.savez_compressed(
    output_path,
    X_spectrograms=X_spectrograms,
    Y_router=Y_router,
    Y_special=Y_special
)

print("\nProcessing complete.")
print(f"Final data shapes saved:\n- Spectrograms: {X_spectrograms.shape}\n- Router Labels:  {Y_router.shape}\n- Special Labels: {Y_special.shape}")
