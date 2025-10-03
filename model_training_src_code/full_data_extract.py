import h5py
import numpy as np


dataset_file = h5py.File("/home/signalsight/.cache/kagglehub/datasets/pinxau1000/radioml2018/versions/2/GOLD_XYZ_OSC.0001_1024.hdf5", "r")

base_modulation_clases = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']

SNRs_in_dataset = [30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10 , -20]
desired_SNRs = [-10, 10, 30]  
signals_per_snr = 4096
samples_per_class = len(SNRs_in_dataset) * signals_per_snr 


family_map = {
    "analog" : ['AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM'],  
    "phase"  : ['BPSK', 'QPSK', '8PSK', '16PSK', '32PSK'],   
    "qam"    : ['4ASK', '8ASK',
                '16QAM', '32QAM', '64QAM', '128QAM'] #'256QAM'
    #"apsk"   : ['16APSK', '32APSK', '64APSK', '128APSK']
}

mod_map = {mod_name: family_name
           for family_name, mod_list in family_map.items()
           for mod_name in mod_list}

signals_per_class = len(SNRs_in_dataset) * signals_per_snr    # 26 × 4096

class_to_id_in_base_list = {}
for i in range(len(base_modulation_clases)):
    modulation_name = base_modulation_clases[i]
    class_to_id_in_base_list[modulation_name] = i

snr_to_offset = {}
for snr in desired_SNRs:
    snr_index = SNRs_in_dataset.index(snr)
    snr_to_offset[snr] = snr_index * signals_per_snr


family_X, family_Y, len_family = {}, {}, {}

for fam_name, mod_list in family_map.items():
    X_list = []
    Y_list = []

    for mod in mod_list:
        base_id = class_to_id_in_base_list[mod] * signals_per_class

        for snr in desired_SNRs:
            start = base_id + snr_to_offset[snr]
            end   = start + signals_per_snr

            X_list.append(dataset_file['X'][start:end])
            Y_list.append(dataset_file['Y'][start:end])

    family_X[fam_name] = np.concatenate(X_list)
    family_Y[fam_name] = np.concatenate(Y_list)
    len_family[fam_name] = len(family_Y[fam_name])
