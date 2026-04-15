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

    # start with synchronization symbol (4410 samples of a 1kHz sinusoid) + zero-pad to let channel settle
    X_t = list(generate_sinusoid(1000)) + list(np.zeros(2000))

    # partition bits into blocks for OOK encoding (one bit per tone)
    num_blocks = len(bits) // K

    for i in range(num_blocks):
        bits_partition = bits[i*K: (i+1)*K]

        gamma = np.sqrt(P * OFDM_LEN / K)

        X_f = bits_partition * gamma

        # construct full frequency spectrum to include conjugates to take DFT
        X_freq = np.zeros(OFDM_LEN, dtype=complex)
        X_freq[1 : K+1] = X_f
        X_freq[OFDM_LEN - K : OFDM_LEN] = np.flip(np.conj(X_f))

        # inverse DFT using FFT
        x_time = fft.ifft(X_freq).real
        
        # add cyclic prefix
        x_time_cp = np.append(x_time[-CP_LEN:], x_time)
        X_t.extend(x_time_cp)
        
    X_t = np.array(X_t)

    tmp = (np.iinfo(np.int32).max*X_t).astype(np.int32)
    wav.write('tx.wav', 44100, tmp)

def generate_sinusoid(freq, amplitude=1.0, num_samples=4410):
    """
    Generate a sinusoid.
    """
    fs = 44100.0
    n = np.arange(num_samples)
    y = amplitude * np.sin(2.0*np.pi*freq* n / fs)

    return y

