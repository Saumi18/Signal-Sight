import numpy as np
import os

GLOBAL_MIN = 6.736997777577654e-16
GLOBAL_MAX = 0.31486624479293823

source_base_dir = 'log_spectrograms'
dest_base_dir = 'final_norm_spectrograms'
os.makedirs(dest_base_dir, exist_ok=True)


print(f"Creating final normalized dataset in '{dest_base_dir}'...")
epsilon = 1e-8


for family_folder in os.listdir(source_base_dir):
    source_dir = os.path.join(source_base_dir, family_folder)
    dest_dir = os.path.join(dest_base_dir, family_folder)
    os.makedirs(dest_dir, exist_ok=True)
    
    if not os.path.isdir(source_dir):
        continue

    # Loop through each log-scaled file
    for filename in os.listdir(source_dir):
        if filename.endswith('.npy'):
            source_file_path = os.path.join(source_dir, filename)
            
            # Load the log-scaled spectrogram
            log_spectrogram = np.load(source_file_path)
            
            # Apply the final global normalization
            norm_spectrogram = (log_spectrogram - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN + epsilon)
            
            # Save the fully normalized file to the new directory
            dest_file_path = os.path.join(dest_dir, filename)
            np.save(dest_file_path, norm_spectrogram)

print('Final dataset creation complete.')
