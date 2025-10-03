import numpy as np
import os

OUTPUT_DIR = 'processed_data'
NUM_SAMPLES_PER_FAMILY = 8000 
JAM_RATIO = 0.5 # 50% of the final dataset will be jammed


def add_cw_jamming(signal_iq, snr_db=-2.5):
    """Adds a high-power Continuous Wave (single tone) jammer."""
    signal = signal_iq[:, 0] + 1j * signal_iq[:, 1]
    
    signal_power = np.mean(np.abs(signal)**2)
    jam_power = signal_power / (10**(snr_db / 10.0))
    
    # Random frequency for the jammer tone
    jam_freq = np.random.uniform(-0.5, 0.5) 
    t = np.arange(len(signal))
    jammer = np.sqrt(jam_power) * np.exp(2j * np.pi * jam_freq * t)
    
    jammed_signal = signal + jammer
    
    return np.stack([jammed_signal.real, jammed_signal.imag], axis=-1)

def add_noise_jamming(signal_iq, snr_db=-5):
    """Adds wideband noise jamming."""
    signal = signal_iq[:, 0] + 1j * signal_iq[:, 1]
    
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / (10**(snr_db / 10.0))
    
    # Generate complex Gaussian noise
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    
    jammed_signal = signal + noise

    return np.stack([jammed_signal.real, jammed_signal.imag], axis=-1)


def add_spoofing_delay(signal_iq, delay_ratio=0.1, alpha=0.7):
    """Simulates a simple meaconing/spoofing attack by adding a delayed version."""
    signal = signal_iq[:, 0] + 1j * signal_iq[:, 1]

    delay_samples = int(len(signal) * delay_ratio)
    spoofed_signal_component = np.zeros_like(signal, dtype=np.complex64)
    
    spoofed_signal_component[delay_samples:] = signal[:-delay_samples] * alpha * np.exp(1j * np.random.uniform(0, 2*np.pi))
    
    final_signal = signal + spoofed_signal_component

    return np.stack([final_signal.real, final_signal.imag], axis=-1)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Mock Data Generation (to replace your import) ---
    print("Generating mock data for demonstration...")
    family_map = {'analog': 0, 'phase': 1, 'qam': 2}
    family_X = {}
    for family in family_map:
        # Generate random I/Q data with shape (NUM_SAMPLES_PER_FAMILY, 1024, 2)
        i_data = np.random.randn(NUM_SAMPLES_PER_FAMILY, 1024)
        q_data = np.random.randn(NUM_SAMPLES_PER_FAMILY, 1024)
        family_X[family] = np.stack([i_data, q_data], axis=-1)
    print("Mock data generated.")
    # --- End of Mock Data Generation ---

    all_clean_signals = []
    # Collect a balanced set of clean signals from all families
    for family in family_map.keys():
        # Ensure we don't request more samples than available
        num_to_sample = min(NUM_SAMPLES_PER_FAMILY, len(family_X[family]))
        indices = np.random.choice(len(family_X[family]), num_to_sample, replace=False)
        all_clean_signals.append(family_X[family][indices])
        
    X_clean = np.vstack(all_clean_signals)
    np.random.shuffle(X_clean) # Shuffle to mix families

    num_jammed = int(len(X_clean) * JAM_RATIO)
    num_normal = len(X_clean) - num_jammed
    
    X_final = []
    Y_final = [] # 0 for normal, 1 for jammed/spoofed

    X_final.extend(X_clean[:num_normal])
    Y_final.extend([0] * num_normal)

    print(f"Collected {len(X_clean)} clean signals.")
    print(f"Keeping {num_normal} as normal.")
    print(f"Generating {num_jammed} jammed/spoofed signals...")

    signals_to_jam = X_clean[num_normal:]
    
    for i, sig in enumerate(signals_to_jam):
        technique = np.random.choice(['cw', 'noise', 'spoof'])
        
        if technique == 'cw':
            jammed_sig = add_cw_jamming(sig)
        elif technique == 'noise':
            jammed_sig = add_noise_jamming(sig)
        else: # spoof
            jammed_sig = add_spoofing_delay(sig)
            
        X_final.append(jammed_sig)
        Y_final.append(1) # Label as jammed

    X_final = np.array(X_final, dtype=np.float32) 
    Y_final = np.array(Y_final, dtype=np.int64)

    output_path = os.path.join(OUTPUT_DIR, 'jamming_raw_signals.npz')
    np.savez_compressed(output_path, X_jammed=X_final, Y_jammed=Y_final)
    
    print(f"\nJamming data generation complete.")
    print(f"Total samples: {len(X_final)}")
    print(f"Final X shape: {X_final.shape}")
    print(f"Data saved to {output_path}")
