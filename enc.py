import numpy as np
import scipy.io.wavfile as wav
import scipy.fft as fft

def enc(bits):
    """
    [bits] is a 1-D numpy array containing 200,000 integers that are either 0 or 1.

    Encodes [bits] and generates a 'tx.wav' file for transmission across CommCloud.
    """

    P = 0.00125 # power constraint
    CP_LEN = 120 # length of cyclic prefix
    OFDM_LEN = 1024 # total length of each OFDM symbol (number of available tones)
    K = 500 # number of usable tones per OFDM symbol

    # start with synchronization symbol (4410 samples of an impulse) + zero-pad to let channel settle
    X_t = list(generate_imp()) + list(np.zeros(2000))

    # partition bits into blocks for OOK encoding (one bit per tone)
    num_blocks = len(bits) // K

    # from lab 1, the channel gain peaks at frequency of 7.5kHz, so we want to center our signals around that
    target_freq = 7500.0
    center_bin = int(np.round(target_freq * OFDM_LEN / 44100))
    
    # get all valid positive frequency bins
    valid_bins = np.arange(1, OFDM_LEN // 2)
    
    # Calculate distance of every bin to the center bin
    distances = np.abs(valid_bins - center_bin)
    
    # sort by distance and pick the K closest bins, then sort indices ascending
    closest_bins = valid_bins[np.argsort(distances)]
    optimal_indices = np.sort(closest_bins[:K])

    for i in range(num_blocks):
        bits_partition = bits[i*K: (i+1)*K]

        gamma = np.sqrt(P * OFDM_LEN / K)

        X_f = bits_partition * gamma

        # construct full frequency spectrum to include conjugates to take DFT
        X_freq = np.zeros(OFDM_LEN, dtype=complex)
        X_freq[optimal_indices] = X_f
        X_freq[OFDM_LEN - optimal_indices] = np.conj(X_f)

        # inverse DFT using FFT
        x_time = fft.ifft(X_freq, norm='ortho').real
        
        # add cyclic prefix
        x_time_cp = np.append(x_time[-CP_LEN:], x_time)
        X_t.extend(x_time_cp)
        
    X_t = np.array(X_t)

    X_t = np.clip(X_t, -1.0, 1.0)

    tmp = (np.iinfo(np.int32).max*X_t).astype(np.int32)
    wav.write('tx.wav', 44100, tmp)

def generate_imp(sample_rate=44100, num_samples=4410):
    """
    Generate an impulse as a numpy array.
    """
    x = np.zeros(num_samples)
    x[0] = 1

    return x
