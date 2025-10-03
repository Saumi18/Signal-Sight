import h5py
import numpy as np

print("--- Running Sanity Check on HDF5 File ---")

HDF5_PATH = "/home/signalsight/.cache/kagglehub/datasets/pinxau1000/radioml2018/versions/2/GOLD_XYZ_OSC.0001_1024.hdf5"
base_modulation_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
SNRs_in_dataset = [30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10, -20]
desired_SNRs = [-10, 10, 30]
signals_per_snr = 4096
signals_per_class = len(SNRs_in_dataset) * signals_per_snr

mods_to_check = ['AM-DSB-WC', 'AM-DSB-SC', 'FM', '32PSK']

try:
    dataset_file = h5py.File(HDF5_PATH, "r")
except Exception as e:
    print(f"Error opening HDF5 file: {e}")
    exit()

print(f"Checking for {len(mods_to_check)} problematic modulations at SNRs: {desired_SNRs}\n")
all_successful = True

for mod_name in mods_to_check:
    print(f"--- Verifying '{mod_name}' ---")
    
    # Get the correct one-hot index for this modulation
    expected_label_index = base_modulation_classes.index(mod_name)
    base_id = expected_label_index * signals_per_class
    
    for snr in desired_SNRs:
        snr_index_in_list = SNRs_in_dataset.index(snr)
        snr_offset = snr_index_in_list * signals_per_snr
        
        start = base_id + snr_offset
        end   = start + signals_per_snr

        # Directly load the labels from this slice in the HDF5 file
        labels_slice = dataset_file['Y'][start:end]
        
        # Check 1: Is the data just all zeros?
        if not np.any(labels_slice):
            print(f"  [SNR {snr:>3}] -> FAILURE: The data slice is all zeros. No samples exist.")
            all_successful = False
            continue
            
        # Check 2: Are the labels correct? The argmax should always be our expected index.
        label_indices_found = np.argmax(labels_slice, axis=1)
        if np.all(label_indices_found == expected_label_index):
            print(f"  [SNR {snr:>3}] -> SUCCESS: Found 4096 correctly labeled samples for '{mod_name}'.")
        else:
            print(f"  [SNR {snr:>3}] -> FAILURE: Data is MISLABELED. Expected all labels to be {expected_label_index} but found others.")
            all_successful = False
    print("-" * 25)

print("\n--- Sanity Check Complete ---")
if all_successful:
    print("✅ All checked modulations appear to exist and are correctly labeled in the HDF5 file.")
    print("   The problem must be a subtle bug in the data extraction loop.")
else:
    print("❌ The HDF5 file is missing or has mislabeled data for the classes listed above at your selected SNRs.")
    print("   This is the reason for the 'support: 0' in your report.")

dataset_file.close()