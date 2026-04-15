import numpy as np
import scipy.io.wavfile as wav
import scipy.linalg as la
from scipy.stats import norm

def dec():
    """
    Opens an 'rx.wav' file, then decodes and returns a 1-D numpy array of 200,000 integer bits.
    """