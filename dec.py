import numpy as np
import scipy.io.wavfile as wav

def dec():

    # ---- Parameters (must match enc.py exactly) ----
    N = 1024           # FFT size
    CP = 120           # cyclic prefix length
    FS = 44100         # sampling rate
    K = 500            # number of data tones per symbol
    NUM_SYMBOLS = 400  # number of data symbols
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

    # ---- Step 3: rebuild the sync symbol (same as enc.py) ----
    freq_sync = np.zeros(N, dtype=complex)
    for k in tone_indices:
        freq_sync[k]     = 1.0
        freq_sync[N - k] = 1.0
    time_sync = np.real(np.fft.ifft(freq_sync))
    sync_symbol = np.concatenate([time_sync[-CP:], time_sync])

    # ---- Step 4: find where the sync symbol starts in the received signal ----
    # we cross-correlate the received signal with the known sync symbol
    # the peak of the correlation tells us where the sync symbol is

    # only use the body of the sync symbol for correlation (not the cyclic prefix)
    sync_body = sync_symbol[CP:]   # length N = 1024

    # cross correlate
    correlation = np.correlate(received, sync_body, mode='full')
    correlation = np.abs(correlation)

    # the peak tells us where sync_body starts in received
    # np.correlate with mode='full' has an offset of len(sync_body)-1
    peak_index = np.argmax(correlation)
    sync_start = peak_index - (len(sync_body) - 1)

    # sync_start is where the first sync symbol body begins
    # but we need to go back CP samples to include the cyclic prefix
    sync_start = sync_start - CP

    # ---- Step 5: skip past both sync symbols to reach data ----
    # we sent 2 sync symbols, each of length SYMBOL_LEN
    data_start = sync_start + 2 * SYMBOL_LEN

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

    # measure average power at our tone indices in the sync symbol
    sync_powers = np.abs(sync1_fft[tone_indices]) ** 2
    avg_on_power = np.mean(sync_powers)

    # threshold is halfway between ON power and zero
    threshold = avg_on_power / 2.0

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

            # OOK decision: above threshold = 1, below = 0
            if power > threshold:
                bits_out[i * K + j] = 1
            else:
                bits_out[i * K + j] = 0

    # return exactly 200,000 bits
    return bits_out[:200000]