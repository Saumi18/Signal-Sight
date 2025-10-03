import numpy as np
from scipy.signal import spectrogram
import os

INPUT_NPZ_PATH = 'processed_data/jamming_raw_signals.npz'
OUTPUT_DIR = 'processed_data'
OUTPUT_FILENAME = 'jamming_spectrograms_and_labels.npz'
FS = 1000      # Sample rate
NPERSEG = 64   # Segment length for FFT
NOVERLAP = 48  # Overlap between segments
EPSILON = 1e-8 # For numerical stability in normalization

def generate_spectrograms(raw_signals_iq):
    """
    Generates and normalizes spectrograms for the entire dataset.
    Uses a two-pass approach for robust global normalization.
    """
    # --- Pass 1: Compute global min/max for normalization ---
    print("Pass 1: Scanning dataset for global min/max values...")
    global_min, global_max = float("inf"), float("-inf")

    for sig_iq in raw_signals_iq:
        # --- FIX: Combine I/Q data into a complex signal ---
        complex_sig = sig_iq[:, 0] + 1j * sig_iq[:, 1]
        
        _, _, Sxx = spectrogram(complex_sig, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
        Sxx_log = np.log1p(np.abs(Sxx))
        global_min = min(global_min, Sxx_log.min())
        global_max = max(global_max, Sxx_log.max())
    print(f"Global min={global_min:.4f}, max={global_max:.4f}")

    # --- Pass 2: Normalize and save spectrograms ---
    print("Pass 2: Generating and normalizing spectrograms...")
    spectrogram_list = []
    for sig_iq in raw_signals_iq:
        # --- FIX: Combine I/Q data into a complex signal ---
        complex_sig = sig_iq[:, 0] + 1j * sig_iq[:, 1]

        _, _, Sxx = spectrogram(complex_sig, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
        Sxx_log = np.log1p(np.abs(Sxx))
        Sxx_norm = (Sxx_log - global_min) / (global_max - global_min + EPSILON)
        spectrogram_list.append(Sxx_norm.astype(np.float32))
    
    return np.array(spectrogram_list)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the augmented raw signals and labels
    print(f"Loading raw signals from {INPUT_NPZ_PATH}...")
    data = np.load(INPUT_NPZ_PATH)
    X_raw_iq = data['X_jammed'] # Data is in I/Q format
    Y_labels = data['Y_jammed']

    # Generate spectrograms from the raw I/Q signals
    X_spectrograms = generate_spectrograms(X_raw_iq)

    # Save the spectrograms and their labels to a new .npz file
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    np.savez_compressed(output_path, 
                        X_spectrograms=X_spectrograms, 
                        Y_jammed_labels=Y_labels)

    print(f"\nSpectrogram generation complete.")
    print(f"Spectrograms shape: {X_spectrograms.shape}")
    print(f"Labels shape: {Y_labels.shape}")
    print(f"Data saved to {output_path}")
