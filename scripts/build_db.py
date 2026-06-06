import os
from pathlib import Path
from src.fingerprint_build import AudioExtract, FingerPrintBuild


if __name__ == "__main__":

    sr = 44100
    n_fft = 1024
    hop_length = n_fft // 2

    
    spec = AudioExtract(sr, n_fft, hop_length)
    peak = FingerPrintBuild()   

