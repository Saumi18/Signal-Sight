import numpy as np

def add_random_spoofing(signal, Fs):
    # Choose a random spoof type
    spoofing_types = ['replay_attack', 'false_flag', 'signal_injection', 'delayed_signal']
    spoof_type = np.random.choice(spoofing_types)

    # Handle (N,2) to complex format
    if signal.ndim == 2 and signal.shape[1] == 2:
        sig_complex = signal[:, 0] + 1j * signal[:, 1]
    else:
        sig_complex = signal

    # Define spoofer power relative to signal power
    spoofer_power_ratio = np.random.uniform(0.1, 1.0)
    signal_power = np.mean(np.abs(sig_complex) ** 2)
    spoofer_power = spoofer_power_ratio * signal_power
    N = len(sig_complex)
    t = np.arange(N) / Fs

    if spoof_type == 'replay_attack':
        # Delayed replay of the original signal
        delay_samples = np.random.randint(int(0.001 * Fs), int(0.05 * Fs)) # 1-50 ms delay
        atten = np.random.uniform(0.5, 1.0)
        replay_signal = np.zeros_like(sig_complex)
        if delay_samples < N:
            replay_signal[delay_samples:] = atten * sig_complex[:-delay_samples]
        spoofed_signal = sig_complex + replay_signal

    elif spoof_type == 'false_flag':
        # Synthetic signal with different modulation (false identity)
        freq = np.random.uniform(0, Fs / 2)
        false_flag = np.sqrt(spoofer_power) * np.exp(2j * np.pi * freq * t)
        noise = (np.random.normal(0, 1, N) + 1j * np.random.normal(0, 1, N))
        noise = np.sqrt(spoofer_power / np.mean(np.abs(noise) ** 2)) * noise * 0.5
        spoofed_signal = sig_complex + false_flag + noise

    elif spoof_type == 'signal_injection':
        burst_len = int(Fs * np.random.uniform(0.001, 0.01))
        if burst_len >= N:
            burst_len = max(1, N - 1)
        start_idx = np.random.randint(0, N - burst_len)
        burst_ampl = np.sqrt(spoofer_power) * np.random.uniform(0.5, 1.0)
        burst = burst_ampl * (np.random.normal(0, 1, burst_len) + 1j * np.random.normal(0, 1, burst_len))
        spoofed_signal = np.copy(sig_complex)
        spoofed_signal[start_idx:start_idx + burst_len] += burst


    elif spoof_type == 'delayed_signal':
        # Add a delayed, phase-shifted version of the signal
        delay_samples = np.random.randint(int(0.002 * Fs), int(0.05 * Fs)) # 2-50 ms delay
        phase = np.random.uniform(0, 2 * np.pi)
        delayed_signal = np.zeros_like(sig_complex)
        atten = np.random.uniform(0.3, 0.9)
        if delay_samples < N:
            delayed_signal[delay_samples:] = atten * sig_complex[:-delay_samples] * np.exp(1j * phase)
        spoofed_signal = sig_complex + delayed_signal

    else:
        spoofed_signal = sig_complex

    # Convert back to (N,2) format if input signal was (N,2)
    if signal.ndim == 2 and signal.shape[1] == 2:
        spoofed_signal = np.column_stack([np.real(spoofed_signal), np.imag(spoofed_signal)])

    spoof_params = {
        'spoof_type': spoof_type,
        'spoofer_power_ratio': spoofer_power_ratio,
    }
    return spoofed_signal, spoof_params
