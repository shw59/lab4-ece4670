import numpy as np
import scipy.io.wavfile as wav

def dec():

    # ---- Parameters (must match enc.py exactly) ----
    N = 1024           # FFT size
    CP = 120           # cyclic prefix length
    FS = 44100         # sampling rate
    K = 350            # number of data tones per symbol
    
    # FIX: Calculate the number of symbols dynamically so it always processes all 200,000 bits
    NUM_SYMBOLS = int(np.ceil(200000 / K))  
    
    SYMBOL_LEN = N + CP  # = 1144 samples per symbol including cyclic prefix

    # ---- Step 1: read rx.wav and convert to float ----
    _, received = wav.read('rx.wav')
    received = received / np.iinfo(np.int32).max

    # ---- Step 2: reconstruct the tone indices (same as enc.py) ----
    center_bin = int(np.round(7500.0 * N / FS))
    valid_bins = np.arange(1, N // 2)
    distances = np.abs(valid_bins - center_bin)
    sorted_by_distance = valid_bins[np.argsort(distances)]
    tone_indices = np.sort(sorted_by_distance[:K])

    # ---- Step 3: rebuild the sync symbol (matching the new PN sequence) ----
    freq_sync = np.zeros(N, dtype=complex)
    np.random.seed(42)
    sync_phases = np.random.choice([1.0, -1.0], size=N)
    
    for k in tone_indices:
        freq_sync[k]     = sync_phases[k]
        freq_sync[N - k] = sync_phases[k]
        
    time_sync = np.real(np.fft.ifft(freq_sync))
    sync_symbol = np.concatenate([time_sync[-CP:], time_sync])

    # ---- Step 4: find where the sync symbol starts in the received signal ----
    double_sync = np.concatenate([sync_symbol, sync_symbol])
    
    # cross correlate
    correlation = np.correlate(received, double_sync, mode='full')
    correlation = np.abs(correlation)

    # the peak tells us where the double_sync sequence starts
    peak_index = np.argmax(correlation)
    
    # FIX: Back off by CP // 2 (60 samples) to absorb the channel delay and center the window
    sync_start = peak_index - (len(double_sync) - 1) - (CP // 2)

    # ---- Step 5: skip past both sync symbols to reach data ----
    # sync_start now points directly to the start of the first sync's cyclic prefix!
    # To reach data, we just add the length of the double_sync we correlated against.
    data_start = sync_start + len(double_sync)

    # ---- Step 6: extract and decode each data symbol ----
    bits_out = np.zeros(NUM_SYMBOLS * K, dtype=int)

    # we need a threshold to decide if a tone is ON or OFF
    # we will set the threshold by measuring power of a sync symbol
    # since we know all tones are ON in the sync symbol, this gives us
    # a reference power level for what "ON" looks like

    # extract first sync symbol body (after cyclic prefix)
    sync1_start = sync_start + CP
    sync1_body = received[sync1_start : sync1_start + N]
    sync1_fft = np.fft.fft(sync1_body)

    # measure exact power at EACH of our tone indices in the sync symbol
    sync_powers = np.abs(sync1_fft[tone_indices]) ** 2
    
    #  list of thresholds for each frequency channel
    thresholds = sync_powers / 2.0

    # now decode each data symbol
    for i in range(NUM_SYMBOLS):

        # find where this symbol starts
        symbol_start = data_start + i * SYMBOL_LEN

        # strip cyclic prefix by skipping first CP samples
        body_start = symbol_start + CP
        body = received[body_start : body_start + N]

        # take FFT
        symbol_fft = np.fft.fft(body)

        # check power at each tone index
        for j in range(K):
            tone_index = tone_indices[j]
            power = np.abs(symbol_fft[tone_index]) ** 2

            # OOK decision: check against the specific threshold for tone j
            if power > thresholds[j]:
                bits_out[i * K + j] = 1
            else:
                bits_out[i * K + j] = 0

    # return exactly 200,000 bits
    return bits_out[:200000]