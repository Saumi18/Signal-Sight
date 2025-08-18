import numpy as np
from full_data_extract import family_X, family_Y, len_family
from jamming_functions import add_random_jamming

def generate_mixed_signal(families, family_X, indices):
    sig = family_X[families[0]][indices]
    for idx, fam in enumerate(families[1:], start=1):
        sig = sig + family_X[fam][indices[idx]]
    return sig

def create_signal_dataset_with_jam(
    pure, mix_2, mix_3, mix_4,
    Fs=1e6, n_jammed_per=1, selective_jam_prob=0.5
):
    family_names = ['analog', 'phase', 'qam', 'apsk']
    dataset_samples = []

    signals_counts = {'pure': pure, 'mix_2': mix_2, 'mix_3': mix_3, 'mix_4': mix_4}

    for target_family in family_names:
        # Pure signals
        for _ in range(signals_counts['pure']):
            idx = np.random.randint(len_family[target_family])
            base_sig = family_X[target_family][idx]
            label = family_Y[target_family][idx]

            #  Clean version
            dataset_samples.append({
                "iq": base_sig,
                "mod_label": label,
                "jam_flag": 0,
                "jam_component_flags": [0],
                "families": [target_family],
                "jammed_families": [],
                "jam_type": None,
                "jam_params": None,
                "spoof_flag": 0,
                "meta": {"type": "pure"}
            })

            #  Jammed versions
            for _ in range(n_jammed_per):
                jammed_sig, jam_params = add_random_jamming(base_sig, Fs)
                dataset_samples.append({
                    "iq": jammed_sig,
                    "mod_label": label,
                    "jam_flag": 1,
                    "jam_component_flags": [1],
                    "families": [target_family],
                    "jammed_families": [target_family],
                    "jam_type": jam_params.get("jam_type", None),
                    "jam_params": jam_params,
                    "spoof_flag": 0,
                    "meta": {"type": "pure"}
                })

        # Mixed signals (selectively jam some components)
        for mix_key, mix_size in [('mix_2', 2), ('mix_3', 3), ('mix_4', 4)]:
            for _ in range(signals_counts[mix_key]):
                mix_fams = np.random.choice(family_names, mix_size, replace=False)
                inds = [np.random.randint(len_family[fam]) for fam in mix_fams]

                components = []
                jam_flags = []
                jammed_fams = []
                mod_labels = []
                jam_types = []
                jam_params_list = []

                for fam, idx in zip(mix_fams, inds):
                    base_sig = family_X[fam][idx]
                    label = family_Y[fam][idx]
                    mod_labels.append(label)
                    if np.random.rand() < selective_jam_prob:
                        jammed_sig, jam_params = add_random_jamming(base_sig, Fs)
                        components.append(jammed_sig)
                        jam_flags.append(1)
                        jammed_fams.append(fam)
                        jam_types.append(jam_params.get("jam_type", None))
                        jam_params_list.append(jam_params)
                    else:
                        components.append(base_sig)
                        jam_flags.append(0)
                        jam_types.append(None)
                        jam_params_list.append(None)

                mixture = sum(components)
                multi_hot_label = np.clip(np.sum(mod_labels, axis=0), 0, 1)

                dataset_samples.append({
                    "iq": mixture,
                    "mod_label": multi_hot_label,
                    "jam_flag": int(any(jam_flags)),
                    "jam_component_flags": jam_flags,
                    "families": list(mix_fams),
                    "jammed_families": jammed_fams,
                    "jam_type": jam_types,
                    "jam_params": jam_params_list,
                    "spoof_flag": 0,
                    "meta": {"type": mix_key}
                })

                
                for _ in range(n_jammed_per-1):
                    new_components = []
                    new_jam_flags = []
                    new_jammed_fams = []
                    new_jam_types = []
                    new_jam_params_list = []
                    for fam, idx in zip(mix_fams, inds):
                        base_sig = family_X[fam][idx]
                        label = family_Y[fam][idx]
                        if np.random.rand() < selective_jam_prob:
                            jammed_sig, jam_params = add_random_jamming(base_sig, Fs)
                            new_components.append(jammed_sig)
                            new_jam_flags.append(1)
                            new_jammed_fams.append(fam)
                            new_jam_types.append(jam_params.get("jam_type", None))
                            new_jam_params_list.append(jam_params)
                        else:
                            new_components.append(base_sig)
                            new_jam_flags.append(0)
                            new_jam_types.append(None)
                            new_jam_params_list.append(None)
                    new_mixture = sum(new_components)
                    dataset_samples.append({
                        "iq": new_mixture,
                        "mod_label": multi_hot_label,
                        "jam_flag": int(any(new_jam_flags)),
                        "jam_component_flags": new_jam_flags,
                        "families": list(mix_fams),
                        "jammed_families": new_jammed_fams,
                        "jam_type": new_jam_types,
                        "jam_params": new_jam_params_list,
                        "spoof_flag": 0,
                        "meta": {"type": mix_key}
                    })

    return dataset_samples

# --- Generate dataset ---
dataset_samples = create_signal_dataset_with_jam(
    pure=100, mix_2=50, mix_3=50, mix_4=50
)
print("Dataset generation complete.")
print(f"Total signals created: {len(dataset_samples)}")


