import h5py
import numpy as np
import json

# --- 1. CONFIGURE THE SIGNAL YOU WANT TO EXTRACT ---
#    Change these two values to get different test signals.
#    Valid modulations: 'BPSK', 'QPSK', '8PSK', 'FM', 'AM-SSB-WC', '16QAM', etc.
#    Valid SNRs: -10, 10, 30 (or others from your `desired_SNRs` list)
# -------------------------------------------------------------
MODULATION_TO_EXTRACT = '16QAM'
SNR_TO_EXTRACT = 10
# -------------------------------------------------------------


# --- Configuration (Copied from your data extraction script) ---
HDF5_PATH = "/home/signalsight/.cache/kagglehub/datasets/pinxau1000/radioml2018/versions/2/GOLD_XYZ_OSC.0001_1024.hdf5"
base_modulation_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
SNRs_in_dataset = [30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10, -20]
signals_per_snr = 4096
signals_per_class = len(SNRs_in_dataset) * signals_per_snr

# --- 2. EXTRACT THE SIGNAL ---
print(f"Extracting one sample of '{MODULATION_TO_EXTRACT}' at {SNR_TO_EXTRACT} dB SNR...")

try:
    with h5py.File(HDF5_PATH, "r") as dataset_file:
        # Calculate the exact position of the signal in the dataset
        mod_index = base_modulation_classes.index(MODULATION_TO_EXTRACT)
        snr_index = SNRs_in_dataset.index(SNR_TO_EXTRACT)
        
        start_index = (mod_index * signals_per_class) + (snr_index * signals_per_snr)
        
        # Extract the first signal from that block (you can change the index for a different sample)
        sample_signal = dataset_file['X'][start_index]
        
        # Convert to a Python list of lists (JSON serializable format)
        signal_as_list = sample_signal.tolist()

        print("\n" + "="*50)
        print(">>> GROUND TRUTH <<<")
        print(f"The actual modulation is: {MODULATION_TO_EXTRACT}")
        print("="*50)

        print("\n--- COPY THE JSON BLOCK BELOW AND PASTE INTO THE WEB UI ---")
        # Print in a compact JSON format
        print(json.dumps(signal_as_list))
        print("-"*(len(MODULATION_TO_EXTRACT) + 30))


except FileNotFoundError:
    print(f"\nERROR: The HDF5 file was not found at the specified path.")
    print(f"       Please check the HDF5_PATH variable in this script.")
except ValueError:
    print(f"\nERROR: The modulation ('{MODULATION_TO_EXTRACT}') or SNR ({SNR_TO_EXTRACT}) was not found.")
    print(f"       Please check that the names are spelled correctly and exist in the dataset lists.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")

