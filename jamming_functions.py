import numpy as np

def add_random_jamming(signal, Fs):
    # Choose a random jam type
    jamming_types = ['tone', 'noise', 'sweep', 'pulsed']
    jam_type = np.random.choice(jamming_types)

    # Handle (N,2) to complex format
    if signal.ndim == 2 and signal.shape[1] == 2:
        sig_complex = signal[:, 0] + 1j * signal[:, 1]
    else:
        sig_complex = signal

    # Define jammer power relative to signal power
    jammer_power_ratio = np.random.uniform(0.1, 1.0)
    signal_power = np.mean(np.abs(sig_complex) ** 2)
    jammer_power = jammer_power_ratio * signal_power

    N = len(sig_complex)
    t = np.arange(N) / Fs

    if jam_type == 'tone':
        f0 = np.random.uniform(0, Fs/2)
        tone = np.sqrt(jammer_power) * np.exp(2j * np.pi * f0 * t)
        jammed_signal = sig_complex + tone

    elif jam_type == 'noise':
        noise = (np.random.normal(0, 1, N) + 1j * np.random.normal(0, 1, N))
        noise = np.sqrt(jammer_power / np.mean(np.abs(noise) ** 2)) * noise
        jammed_signal = sig_complex + noise

    elif jam_type == 'sweep':
        f_start = np.random.uniform(0, Fs/4)
        f_end = np.random.uniform(Fs/4, Fs/2)
        sweep = np.sqrt(jammer_power) * np.exp(2j * np.pi * ((f_end - f_start) / (2 * N / Fs) * t ** 2 + f_start * t))
        jammed_signal = sig_complex + sweep

    elif jam_type == 'pulsed':
        pulse_len = int(Fs * 1e-3)
        num_pulses = np.random.randint(1, 5)
        jammed_signal = np.copy(sig_complex)
        for _ in range(num_pulses):
            start_idx = np.random.randint(0, N - pulse_len)
            amplitude = np.sqrt(jammer_power) * np.random.uniform(0.5, 1.0)
            pulse = amplitude * (np.random.normal(0, 1, pulse_len) + 1j * np.random.normal(0, 1, pulse_len))
            jammed_signal[start_idx:start_idx+pulse_len] += pulse

    else:
        jammed_signal = sig_complex

    # Convert back to (N,2) format if input signal was (N,2)
    if signal.ndim == 2 and signal.shape[1] == 2:
        jammed_signal = np.column_stack([np.real(jammed_signal), np.imag(jammed_signal)])

    jam_params = {
        'jam_type': jam_type,
        'jammer_power_ratio': jammer_power_ratio,
    }
    return jammed_signal, jam_params

