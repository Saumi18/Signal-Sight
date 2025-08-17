import numpy as np
from jamming_functions import add_random_jamming
from full_data_extract import family_X, family_Y, len_family

def generate_mixed_signal(families, family_X, indices):
    sig = family_X[families[0]][indices]
    for idx, fam in enumerate(families[1:]):
        sig = sig + family_X[fam][indices[idx + 1]]
    return sig

def create_dataset_with_jamming(pure, mix_2, mix_3, mix_4, Fs=1e6, n_jammed_per=1):
    family_names = ['analog', 'phase', 'qam', 'apsk']
    X_data, Y_data, meta_data = [], [], []
    
    signals_count = {'pure': pure, 'mix_2': mix_2, 'mix_3': mix_3, 'mix_4': mix_4}
    
    for target_family in family_names:
        # Pure signals
        for _ in range(signals_count['pure']):
            idx = np.random.randint(len_family[target_family])
            sig_clean = family_X[target_family][idx]
            label_clean = family_Y[target_family][idx]
            
            # Clean variants
            X_data.append(sig_clean)
            Y_data.append(label_clean)
            meta_data.append({'type':'pure', 'jammed':0, 'family':target_family})
            
            # Jammed variants
            for _ in range(n_jammed_per):
                jammed_sig, jam_params = add_random_jamming(sig_clean, Fs)
                X_data.append(jammed_sig)
                Y_data.append(label_clean)
                jam_meta = {'type':'pure', 'jammed':1, 'family':target_family}
                jam_meta.update(jam_params)
                meta_data.append(jam_meta)
        
        # Mix 2 families
        for _ in range(signals_count['mix_2']):
            mix_fams = np.random.choice(family_names, 2, replace=False)
            inds = [np.random.randint(len_family[fam]) for fam in mix_fams]
            sig = generate_mixed_signal(mix_fams, family_X, inds)
            label = family_Y[mix_fams[0]][inds]
            
            X_data.append(sig)
            Y_data.append(label)
            meta_data.append({'type':'mix_2', 'jammed':0, 'families':list(mix_fams)})
            
            for _ in range(n_jammed_per):
                jammed_sig, jam_params = add_random_jamming(sig, Fs)
                X_data.append(jammed_sig)
                Y_data.append(label)
                jam_meta = {'type':'mix_2', 'jammed':1, 'families':list(mix_fams)}
                jam_meta.update(jam_params)
                meta_data.append(jam_meta)
        
        # Mix 3 families
        for _ in range(signals_count['mix_3']):
            mix_fams = np.random.choice(family_names, 3, replace=False)
            inds = [np.random.randint(len_family[fam]) for fam in mix_fams]
            sig = generate_mixed_signal(mix_fams, family_X, inds)
            label = family_Y[mix_fams[0]][inds]
            
            X_data.append(sig)
            Y_data.append(label)
            meta_data.append({'type':'mix_3', 'jammed':0, 'families':list(mix_fams)})
            
            for _ in range(n_jammed_per):
                jammed_sig, jam_params = add_random_jamming(sig, Fs)
                X_data.append(jammed_sig)
                Y_data.append(label)
                jam_meta = {'type':'mix_3', 'jammed':1, 'families':list(mix_fams)}
                jam_meta.update(jam_params)
                meta_data.append(jam_meta)
        
        # Mix 4 families
        for _ in range(signals_count['mix_4']):
            mix_fams = np.random.choice(family_names, 4, replace=False)
            inds = [np.random.randint(len_family[fam]) for fam in mix_fams]
            sig = generate_mixed_signal(mix_fams, family_X, inds)
            label = family_Y[mix_fams[0]][inds]
            
            X_data.append(sig)
            Y_data.append(label)
            meta_data.append({'type':'mix_4', 'jammed':0, 'families':list(mix_fams)})
            
            for _ in range(n_jammed_per):
                jammed_sig, jam_params = add_random_jamming(sig, Fs)
                X_data.append(jammed_sig)
                Y_data.append(label)
                jam_meta = {'type':'mix_4', 'jammed':1, 'families':list(mix_fams)}
                jam_meta.update(jam_params)
                meta_data.append(jam_meta)
    
    return np.array(X_data), np.array(Y_data), meta_data
