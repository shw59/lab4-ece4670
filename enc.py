import numpy as np
import scipy.io.wavfile as wav
import scipy.linalg as la
from scipy.stats import norm

def enc(bits):
    """
    [bits] is a 1-D numpy array containing 200,000 integers that are either 0 or 1.

    Encodes [bits] and generates a 'tx.wav' file for transmission across CommCloud.
    """

    P = 0.00125 # power constraint
    CP_LEN = 100 # length of cyclic prefix
    OFDM_LEN = 1024 # length of each OFDM symbol

    tmp = (np.iinfo(np.int32).max*X).astype(np.int32)
    wav.write('tx.wav', 44100, tmp)



