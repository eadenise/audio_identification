import librosa
import os
import numpy as np
from scipy.ndimage import maximum_filter
from scripts.config import *


class AudioExtract:
    def __init__(self, sr, n_fft, hop_length):
        self.sr: int = sr
        self.n_fft: int = n_fft
        self.hop_length: int = hop_length

    def extract_audio_spec(self, path: str) -> dict:
        audio_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.wav', '.mp3'))]

        spectrogram_dict = {}

        for audio_file in audio_files:
            fname = os.path.basename(audio_file)
            # Map file name to each spectrogram
            spectrogram = self.get_single_file(audio_file)
            spectrogram_dict[fname] = spectrogram
        
        return spectrogram_dict


        
    def get_single_file(self, audio_file:str) -> np.ndarray:
        y, orig_sr = librosa.load(audio_file, sr=self.sr)

        # resample
        if orig_sr != self.sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.sr)

        spectrogram = np.abs(librosa.stft(y=y, n_fft=self.n_fft, hop_length=self.hop_length, window='hann', center=False))
        log_mag_spec = np.log(spectrogram + 1e-10)
        return log_mag_spec

    
class FingerPrintBuild:
    def __init__(self, spectrogram):
        self.spectrogram = spectrogram

    def get_salient_peaks(self, sr: int, n_fft: int, hop_length: int, target_peak_s: int, window_sec: int) -> list:
        frames_per_sec = sr // hop_length
        block_size = int(round(window_sec * frames_per_sec))
        num_bins, num_frames = self.spectrogram.shape

        freq_res = sr / n_fft
        time_res = hop_length / sr

        target_hz = 200.0 # exclude below 200 Hz
        target_sec = 0.3 # suppress sustained notes

        # Find STFT resolution
        n_freq_bins = int(round(target_hz / freq_res))
        n_time_frames = int(round(target_sec / time_res))
        # Force dimensions to be odd numbers for a centered peak-picking filter
        n_freq_bins = n_freq_bins if n_freq_bins % 2 != 0 else n_freq_bins + 1
        n_time_frames = n_time_frames if n_time_frames % 2 != 0 else n_time_frames + 1


        selected_peaks = [] # List of tuples: bin index and frame index

        # Iterate through frames
        for start_frame in range(0, num_frames, block_size):
            end_frame = min(start_frame + block_size, num_frames)

            # Extract the current block (Bins x Time Frames in Window)
            block = self.spectrogram[:, start_frame:end_frame]
            # Extract local max peaks per frame
            min_thresh = np.mean(block) + np.std(block)
            local_max_peak = maximum_filter(block, size=(n_freq_bins, n_time_frames))
            peak_mask = (block == local_max_peak) & (block > min_thresh)

            bin_idx, frame_offsets = np.where(peak_mask)
            magnitudes = block[bin_idx, frame_offsets]

            # Select top N salient peaks per 1s block
            if len(magnitudes) > 0:
            # Get indices that sort magnitudes in descending order
                top_k_indices = np.argsort(magnitudes)[::-1][:target_peak_s]

                for idx in top_k_indices:
                    absolute_bin = int(bin_idx[idx])
                    # Map relative block frame back to the global spectrogram frame
                    absolute_frame = int(start_frame + frame_offsets[idx])
                    selected_peaks.append((absolute_bin, absolute_frame))

        return selected_peaks

    def generate_hash(self, peaks: list[tuple[int, int]]) -> list[tuple[str, int]]:

        F = 5 # fan out factor/ how many points are paired with the anchor point
        target_zone_size = 15

        # sort peaks by time frame
        peaks.sort(key=lambda x: x[1])

        fingerprints = []

        for i in range(len(peaks)): 
            f_anchor, t_anchor = peaks[i]

            # get next target zone size peaks after anchor
            zone_end = min(i + 1 + target_zone_size, len(peaks))
            targets = peaks[i + 1 : zone_end]

            
            # Takes on F that are closest in time to the anchor 
            for f_target, t_target in targets[:F]:
                # minimal time distance computation
                delta_t = t_target - t_anchor

                # create hash: 32 bit unsigned int
                hash_val = (int(f_anchor & 0x3FFF) << 18) | (int(f_target & 0x3FF) << 8) | int(delta_t & 0xFF)
                hash_val_32 = np.uint32(hash_val)
                fingerprints.append((int(hash_val_32), int(t_anchor)))

        return fingerprints

