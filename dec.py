import numpy as np
import scipy.io.wavfile as wav
import scipy.linalg as la
from scipy.stats import norm

def dec():
    """
    Opens an 'rx.wav' file, then decodes and returns a 1-D numpy array of 200,000 integer bits.
    """
    P = 0.00125 # power constraint
    CP_LEN = 120 # length of cyclic prefix
    OFDM_LEN = 1024 # total length of each OFDM symbol (number of available tones)
    K = 500 # number of usable tones per OFDM symbol


    Fs, Y = wav.read('rx.wav')
    Y = (Y/np.iinfo(np.int32).max)[int(0.1 * Fs):]


def generate_sinusoid(freq, amplitude=1.0, num_samples=4410):
    """
    Generate a sinusoid.
    """
    fs = 44100.0
    n = np.arange(num_samples)
    y = amplitude * np.sin(2.0*np.pi*freq* n / fs)

    return y
