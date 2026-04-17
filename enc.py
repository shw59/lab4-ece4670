import numpy as np
import scipy.io.wavfile as wav


def enc(bits):

    # Parameters
    N = 1024 # size of each OFDM symbol
    CP = 256 # cyclic prefix length
    FS = 44100 # sampling rate in Hz
    K = 350 # number of tones carrying data per symbol
    TARGET_POWER = 0.00125  # power constraint from lab

    # Find the K best tone indices centered around 7500 Hz
    center_bin = int(np.round(7500.0 * N / FS))   # = 174

    # all valid positive frequency bins
    valid_bins = np.arange(1, N // 2)

    # sort bins by distance to center bin, pick closest K, then sort ascending
    distances = np.abs(valid_bins - center_bin)
    sorted_by_distance = valid_bins[np.argsort(distances)]
    tone_indices = np.sort(sorted_by_distance[:K])

    # Figure out how many OFDM symbols we need
    bits_per_symbol = K
    num_symbols = int(np.ceil(len(bits) / bits_per_symbol))   # = 400

    # pad bits just in case
    bits_padded = np.zeros(num_symbols * bits_per_symbol, dtype=int)
    bits_padded[0:len(bits)] = bits

    # build the sync symbol
    freq_sync = np.zeros(N, dtype=complex)
    
    # use a Pseudo-Noise (PN) sequence to prevent the massive time-domain spike
    np.random.seed(42) # Seed ensures encoder and decoder generate the exact same phases
    sync_phases = np.random.choice([1.0, -1.0], size=N)
    
    for k in tone_indices:
        freq_sync[k]     = sync_phases[k]
        freq_sync[N - k] = sync_phases[k]

    # convert to time domain
    time_sync = np.real(np.fft.ifft(freq_sync))

    # add cyclic prefix
    sync_symbol = np.concatenate([time_sync[-CP:], time_sync])

    # build all data symbols
    all_symbols = []

    for i in range(num_symbols):

        # grab the 350 bits for this symbol
        start = i * bits_per_symbol
        end   = start + bits_per_symbol
        bits_this_symbol = bits_padded[start:end]

        # build frequency domain array for this symbol
        freq_data = np.zeros(N, dtype=complex)

        for j in range(K):
            tone_index = tone_indices[j]
            bit_value  = bits_this_symbol[j]

            # apply the PN sequence phase to spread the energy and avoid the t=0 spike
            phase = sync_phases[tone_index]
            freq_data[tone_index]     = float(bit_value) * phase
            freq_data[N - tone_index] = float(bit_value) * phase

        # convert to time domain
        time_data = np.real(np.fft.ifft(freq_data))

        # add cyclic prefix
        symbol_with_cp = np.concatenate([time_data[-CP:], time_data])
        all_symbols.append(symbol_with_cp)

    # concatenate 2 sync symbols + 400 data symbols
    signal = np.concatenate([sync_symbol, sync_symbol] + all_symbols)

    # scale to meet power constraint
    current_power = np.mean(signal ** 2)
    scale_factor  = np.sqrt(TARGET_POWER / current_power)
    signal        = signal * scale_factor

    # clip to ensure it's between -1.0 and 1.0
    signal = np.clip(signal, -1.0, 1.0)

    # write tx.wav
    signal_int32 = (np.iinfo(np.int32).max * signal).astype(np.int32)
    wav.write('tx.wav', FS, signal_int32)


    