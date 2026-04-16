import numpy as np
import scipy.io.wavfile as wav
import subprocess
import shutil
from enc import enc
from dec import dec

def run_and_check_figure_of_merit(channel='audio0', use_ccplay=True):
    """
    Automatically runs the encoder, simulates the channel, runs the decoder, 
    calculates bit errors, and evaluates the Lab 4 Figure of Merit.
    """
    # 1. Generate exactly 200,000 random data bits (0s and 1s)
    print("Generating 200,000 random bits...")
    original_bits = np.random.randint(0, 2, 200000)
    
    # 2. Run the encoder to produce tx.wav
    print("Running encoder...")
    enc(original_bits)
    
    # 3. Simulate the channel to produce rx.wav
    if use_ccplay:
        print(f"Running ccplay on {channel}...")
        # The lab specifies L0 and L1 should be random integers between 50 and 5000
        L0 = np.random.randint(50, 5001)
        L1 = np.random.randint(50, 5001)
        
        # Call the ccplay script using the absolute path provided in the primer
        ccplay_cmd = [
            "/classes/ece4670/ccplay/ccplay", 
            "--prepause", str(L0), 
            "--postpause", str(L1), 
            "--channel", channel, 
            "tx.wav", "rx.wav"
        ]
        
        # Execute the command and wait for it to finish
        subprocess.run(ccplay_cmd, check=True)
    else:
        print("Skipping ccplay. Simulating a perfect channel (identity operator)...")
        # Copy tx.wav directly to rx.wav to simulate a perfect, noise-free channel
        shutil.copyfile('tx.wav', 'rx.wav')
        
    # 4. Run the decoder to read rx.wav and extract bits
    print("Running decoder...")
    decoded_bits = dec()
    
    # 5. Calculate Bit Errors (N) by directly comparing the arrays
    N = np.sum(original_bits != decoded_bits)
    
    # 6. Read tx.wav to calculate Power (P) and Data Rate (R)
    fs, X_tx_int = wav.read('tx.wav')
    X_tx = X_tx_int / np.iinfo(np.int32).max
    P = np.mean(X_tx**2)
    
    duration_sec = len(X_tx) / fs
    R = 200000 / duration_sec
    
    # 7. Calculate Figure of Merit
    numerator = min(R, 10000) * ((1 - N / 10000)**10)
    denominator = max(1, 800 * P)
    fom = numerator / denominator
    
    # Print the results
    print(f"\n--- Lab 4 System Performance ---")
    print(f"Channel         : {'Perfect (tx.wav -> rx.wav)' if not use_ccplay else channel}")
    print(f"tx.wav Duration : {duration_sec:.4f} seconds")
    print(f"Data Rate (R)   : {R:.2f} bps")
    print(f"Avg Power (P)   : {P:.6f}")
    print(f"Total Errors (N): {N} out of 200,000")
    print(f"\n>>> FIGURE OF MERIT: {fom:.2f} <<<")
    
    # Warn if the soft power constraint is violated
    if P > 0.00125:
        print("\nWARNING: Your average power P exceeds 0.00125!")
        print("The max(1, 800*P) denominator is actively penalizing your score.")

if __name__ == "__main__":
    # Change use_ccplay to False if you want to test your baseline logic first
    run_and_check_figure_of_merit(channel='audio0', use_ccplay=False)