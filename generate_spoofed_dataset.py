import numpy as np
from full_data_extract import family_X, family_Y, len_family
from spoofed_functions import add_random_spoofing  # Ensure this imports the spoofing function correctly
import os

def generate_mixed_signal(families, family_X, indices):
    sig = family_X[families[0]][indices]
    for idx, fam in enumerate(families[1:], start=1):
        sig = sig + family_X[fam][indices[idx]]
    return sig

def create_signal_dataset_with_spoof(
    pure, mix_2, mix_3, mix_4,
    Fs=1e6, n_spoofed_per=1, selective_spoof_prob=0.5
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

            # Clean version
            dataset_samples.append({
                "iq": base_sig,
                "mod_label": label,
                "spoof_flag": 0,
                "spoof_component_flags": [0],
                "families": [target_family],
                "spoofed_families": [],
                "spoof_type": None,
                "spoof_params": None,
                "jam_flag": 0,
                "meta": {"type": "pure"}
            })

            # Spoofed versions
            for _ in range(n_spoofed_per):
                spoofed_sig, spoof_params = add_random_spoofing(base_sig, Fs)
                dataset_samples.append({
                    "iq": spoofed_sig,
                    "mod_label": label,
                    "spoof_flag": 1,
                    "spoof_component_flags": [1],
                    "families": [target_family],
                    "spoofed_families": [target_family],
                    "spoof_type": spoof_params.get("spoof_type", None),
                    "spoof_params": spoof_params,
                    "jam_flag": 0,
                    "meta": {"type": "pure"}
                })

        # Mixed signals (selectively spoof some components)
        for mix_key, mix_size in [('mix_2', 2), ('mix_3', 3), ('mix_4', 4)]:
            for _ in range(signals_counts[mix_key]):
                mix_fams = np.random.choice(family_names, mix_size, replace=False)
                inds = [np.random.randint(len_family[fam]) for fam in mix_fams]

                components = []
                spoof_flags = []
                spoofed_fams = []
                mod_labels = []
                spoof_types = []
                spoof_params_list = []

                for fam, idx in zip(mix_fams, inds):
                    base_sig = family_X[fam][idx]
                    label = family_Y[fam][idx]
                    mod_labels.append(label)

                    if np.random.rand() < selective_spoof_prob:
                        spoofed_sig, spoof_params = add_random_spoofing(base_sig, Fs)
                        components.append(spoofed_sig)
                        spoof_flags.append(1)
                        spoofed_fams.append(fam)
                        spoof_types.append(spoof_params.get("spoof_type", None))
                        spoof_params_list.append(spoof_params)
                    else:
                        components.append(base_sig)
                        spoof_flags.append(0)
                        spoof_types.append(None)
                        spoof_params_list.append(None)

                mixture = sum(components)
                multi_hot_label = np.clip(np.sum(mod_labels, axis=0), 0, 1)

                dataset_samples.append({
                    "iq": mixture,
                    "mod_label": multi_hot_label,
                    "spoof_flag": int(any(spoof_flags)),
                    "spoof_component_flags": spoof_flags,
                    "families": list(mix_fams),
                    "spoofed_families": spoofed_fams,
                    "spoof_type": spoof_types,
                    "spoof_params": spoof_params_list,
                    "jam_flag": 0,
                    "meta": {"type": mix_key}
                })

                for _ in range(n_spoofed_per - 1):
                    new_components = []
                    new_spoof_flags = []
                    new_spoofed_fams = []
                    new_spoof_types = []
                    new_spoof_params_list = []
                    for fam, idx in zip(mix_fams, inds):
                        base_sig = family_X[fam][idx]
                        label = family_Y[fam][idx]
                        if np.random.rand() < selective_spoof_prob:
                            spoofed_sig, spoof_params = add_random_spoofing(base_sig, Fs)
                            new_components.append(spoofed_sig)
                            new_spoof_flags.append(1)
                            new_spoofed_fams.append(fam)
                            new_spoof_types.append(spoof_params.get("spoof_type", None))
                            new_spoof_params_list.append(spoof_params)
                        else:
                            new_components.append(base_sig)
                            new_spoof_flags.append(0)
                            new_spoof_types.append(None)
                            new_spoof_params_list.append(None)
                    new_mixture = sum(new_components)
                    dataset_samples.append({
                        "iq": new_mixture,
                        "mod_label": multi_hot_label,
                        "spoof_flag": int(any(new_spoof_flags)),
                        "spoof_component_flags": new_spoof_flags,
                        "families": list(mix_fams),
                        "spoofed_families": new_spoofed_fams,
                        "spoof_type": new_spoof_types,
                        "spoof_params": new_spoof_params_list,
                        "jam_flag": 0,
                        "meta": {"type": mix_key}
                    })

    return dataset_samples

def organize_spoofed_signals_by_family(dataset_samples):
    family_names = ['analog', 'phase', 'qam', 'apsk']
    X_spoofed = {family: [] for family in family_names}

    for sample in dataset_samples:
        if sample['spoof_flag'] == 1:
            if len(sample['families']) == 1:
                family = sample['families'][0]
                X_spoofed[family].append(sample['iq'])
            else:
                primary_family = sample['families'][0]
                X_spoofed[primary_family].append(sample['iq'])

    for family in family_names:
        X_spoofed[family] = np.array(X_spoofed[family])
        print(f"X_spoofed['{family}'] shape: {X_spoofed[family].shape}")

    return X_spoofed

# Generate spoofed dataset with balanced counts, similar proportions as clean dataset
dataset_samples = create_signal_dataset_with_spoof(
    pure=9832,
    mix_2=16384,
    mix_3=19660,
    mix_4=19660,
    n_spoofed_per=1,
    selective_spoof_prob=0.5
)

X_spoofed = organize_spoofed_signals_by_family(dataset_samples)

print("Spoofed dataset generation complete.")
print(f"Total signals created: {len(dataset_samples)}")
print(f"Spoofed signals by family:")
for family, signals in X_spoofed.items():
    print(f"  {family}: {len(signals)} spoofed signals")
