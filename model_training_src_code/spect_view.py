import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
NPZ_DATA_PATH = 'processed_data/jamming_spectrograms_and_labels.npz'
VISUALIZATION_DIR = 'visualizations'
OUTPUT_FILENAME = 'spectrogram_comparison.png'

def view_sample_spectrograms(npz_path):
    """
    Loads spectrogram data and saves a plot comparing one sample of a 'normal'
    signal and one sample of a 'jammed/spoofed' signal.
    """
    if not os.path.exists(npz_path):
        print(f"Error: Data file not found at '{npz_path}'")
        print("Please run the jam_data_aug.py and jam_spect_gen.py scripts first.")
        return

    print(f"Loading spectrograms from {npz_path}...")
    data = np.load(npz_path)
    X = data['X_spectrograms']
    Y = data['Y_jammed_labels']
    
    # Find the first index for each class
    try:
        normal_idx = np.where(Y == 0)[0][0]
        jammed_idx = np.where(Y == 1)[0][0]
    except IndexError:
        print("Error: Could not find samples for both 'normal' and 'jammed' classes.")
        print("Please check your data generation process.")
        return

    print(f"Found a 'normal' sample at index {normal_idx}")
    print(f"Found a 'jammed' sample at index {jammed_idx}")

    # Get the spectrograms
    normal_spectrogram = X[normal_idx]
    jammed_spectrogram = X[jammed_idx]

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Normal Spectrogram
    im1 = axes[0].imshow(normal_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Sample: Normal Signal Spectrogram')
    axes[0].set_xlabel('Time Segments')
    axes[0].set_ylabel('Frequency Bins')
    fig.colorbar(im1, ax=axes[0], label='Normalized Power')
    
    # Plot Jammed Spectrogram
    im2 = axes[1].imshow(jammed_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Sample: Jammed/Spoofed Signal Spectrogram')
    axes[1].set_xlabel('Time Segments')
    axes[1].set_ylabel('Frequency Bins')
    fig.colorbar(im2, ax=axes[1], label='Normalized Power')
    
    plt.tight_layout()
    plt.suptitle('Spectrogram Comparison', fontsize=16, y=1.02)
    
    # --- Save the plot to a file ---
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    output_path = os.path.join(VISUALIZATION_DIR, OUTPUT_FILENAME)
    plt.savefig(output_path)
    plt.close(fig) # Close the figure to free up memory

    print(f"\nSpectrogram comparison plot saved to: {output_path}")


if __name__ == "__main__":
    view_sample_spectrograms(NPZ_DATA_PATH)

